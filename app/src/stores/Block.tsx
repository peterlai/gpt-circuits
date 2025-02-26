import { atom } from "jotai";
import { selectAtom } from "jotai/utils";

import { atomWithQuery } from "jotai-tanstack-query";
import { SAMPLES_ROOT_URL } from "../views/App/urls";
import { modelIdAtom, sampleDataAtom, sampleIdAtom, versionAtom } from "./Graph";
import { SampleData } from "./Sample";
import { SelectionState, selectionStateAtom } from "./Selection";

// Represents a transformer block (or embedding) in the ablation graph
class BlockData {
  public tokenOffset: number;
  public layerIdx: number;
  public features: { [key: string]: BlockFeatureData };

  constructor(tokenOffset: number, layerIdx: number) {
    this.tokenOffset = tokenOffset;
    this.layerIdx = layerIdx;
    this.features = {};
  }

  get key() {
    return BlockData.getKey(this.tokenOffset, this.layerIdx);
  }

  static getKey(tokenOffset: number, layerIdx: number) {
    return `${tokenOffset}.${layerIdx}`;
  }

  static compare(a: BlockData, b: BlockData) {
    if (a.layerIdx === b.layerIdx) {
      return -(a.tokenOffset - b.tokenOffset);
    }
    return a.layerIdx - b.layerIdx;
  }
}

// Represents a feature on a specific layer and token offset
class BlockFeatureData {
  public tokenOffset: number;
  public layerIdx: number;
  public featureId: number;
  public activation: number;
  public normalizedActivation: number;
  public ablatedBy: { [key: string]: UpstreamAblationData };
  public groupAblations: { [key: number]: number }; // Maps upstream token offset to weight

  constructor(
    tokenOffset: number,
    layerIdx: number,
    featureId: number,
    activation: number,
    normalizedActivation: number
  ) {
    this.tokenOffset = tokenOffset;
    this.layerIdx = layerIdx;
    this.featureId = featureId;
    this.activation = activation;
    this.normalizedActivation = normalizedActivation;
    this.ablatedBy = {};
    this.groupAblations = {};
  }

  get key() {
    return BlockFeatureData.getKey(this.tokenOffset, this.layerIdx, this.featureId);
  }

  // Returns DOM ID
  get elementId() {
    return BlockFeatureData.getElementId(this.key);
  }

  static getKey(tokenOffset: number, layerIdx: number, featureId: number) {
    return `${tokenOffset}.${layerIdx}.${featureId}`;
  }

  static getElementId(key: string) {
    return "Feature_" + key.replace(/\./g, "-");
  }
}

// Represents an upstream ablation of a feature by another feature
class UpstreamAblationData {
  public tokenOffset: number;
  public layerIdx: number;
  public featureId: number;
  public weight: number;

  constructor(tokenOffset: number, layerIdx: number, featureId: number, weight: number) {
    this.tokenOffset = tokenOffset;
    this.layerIdx = layerIdx;
    this.featureId = featureId;
    this.weight = weight;
  }

  get key() {
    return UpstreamAblationData.getKey(this.tokenOffset, this.layerIdx, this.featureId);
  }

  static getKey(tokenOffset: number, layerIdx: number, featureId: number) {
    return `${tokenOffset}.${layerIdx}.${featureId}`;
  }
}

// Block data as parsed from ablation graph
const blocksAtom = atom((get) => {
  const blocks: { [key: string]: BlockData } = {};
  const data = get(sampleDataAtom).data;
  const ablationGraph = data?.graph ?? {};
  const activations: { [key: string]: number } = data?.activations ?? {};
  const normalizedActivations: { [key: string]: number } = data?.normalizedActivations ?? {};
  const groupAblationGraph: { [key: string]: [string, number][] } = data?.blockImportance ?? {};

  // Gather all unique feature keys
  const featureKeys = new Set<string>();
  for (const [downstreamKey, ablations] of Object.entries(ablationGraph)) {
    featureKeys.add(downstreamKey);
    for (const [upstreamKey] of ablations as [string, number][]) {
      featureKeys.add(upstreamKey);
    }
  }

  // Create block and feature data
  for (const featureKey of Array.from(featureKeys.values())) {
    const [tokenOffset, layerIdx, featureId] = featureKey.split(".").map(Number);
    const blockKey = BlockData.getKey(tokenOffset, layerIdx);
    const blockData = blocks[blockKey] || new BlockData(tokenOffset, layerIdx);
    blocks[blockKey] = blockData;

    const activation = activations[featureKey] || 0;
    const normalizedActivation = normalizedActivations[featureKey] || 0;
    const featureData =
      blockData.features[featureKey] ||
      new BlockFeatureData(tokenOffset, layerIdx, featureId, activation, normalizedActivation);
    blockData.features[featureKey] = featureData;
  }

  // Add ablations to features
  for (const [downstreamKey, ablations] of Object.entries(ablationGraph)) {
    // Get feature
    const [tokenOffset, layerIdx] = downstreamKey.split(".").map(Number);
    const blockKey = BlockData.getKey(tokenOffset, layerIdx);
    const featureData = blocks[blockKey].features[downstreamKey];

    // Add ablations to feature
    for (const [upstreamKey, weight] of ablations as [string, number][]) {
      const [upstreamTokenOffset, upstreamLayerIdx, upstreamFeatureId] = upstreamKey
        .split(".")
        .map(Number);
      const ablationKey = UpstreamAblationData.getKey(
        upstreamTokenOffset,
        upstreamLayerIdx,
        upstreamFeatureId
      );
      featureData.ablatedBy[ablationKey] = new UpstreamAblationData(
        upstreamTokenOffset,
        upstreamLayerIdx,
        upstreamFeatureId,
        weight
      );
    }
  }

  // Add group ablations to features
  for (const [downstreamKey, groupAblations] of Object.entries(groupAblationGraph)) {
    // Get feature
    const [tokenOffset, layerIdx] = downstreamKey.split(".").map(Number);
    const blockKey = BlockData.getKey(tokenOffset, layerIdx);
    const featureData = blocks[blockKey].features[downstreamKey];

    // Add group ablations to feature
    for (const [blockKey, weight] of groupAblations as [string, number][]) {
      const upstreamTokenOffset = Number(blockKey.split(".")[0]);
      featureData.groupAblations[upstreamTokenOffset] = weight;
    }
  }

  return blocks;
});

// Represents modifiers to apply to a specific block
class BlockModifier {
  public isEmphasized: boolean = false;
  public isHovered: boolean = false;
  public isSelected: boolean = false;
  public isFocused: boolean = false;

  constructor(block: BlockData, selectionState: SelectionState) {
    // Show outline if the block is at the hovered upstream offset
    this.isEmphasized =
      selectionState.focusedFeature?.layerIdx === block.layerIdx + 1 &&
      selectionState.hoveredUpstreamOffset === block.tokenOffset;
    // Update selection state
    this.isHovered = block.key === selectionState.hoveredBlock?.key;
    this.isSelected = block.key === selectionState.selectedBlock?.key;
    this.isFocused = block.key === selectionState.focusedBlock?.key;
  }

  // Used to avoid re-rendering when the modifier hasn't changed
  static areEqual(a: BlockModifier, b: BlockModifier) {
    return (
      a.isEmphasized === b.isEmphasized &&
      a.isHovered === b.isHovered &&
      a.isSelected === b.isSelected &&
      a.isFocused === b.isFocused
    );
  }
}

// Creates a block modifier atom for a specific block
function createBlockModifierAtom(block: BlockData) {
  return selectAtom(
    selectionStateAtom,
    (selectionState) => {
      return new BlockModifier(block, selectionState);
    },
    BlockModifier.areEqual
  );
}

// Represents supplementary data for a specific block
class BlockProfile {
  public maxActivation: number = 0;
  public samples: SampleData[] = [];

  constructor(data: { [key: string]: number | [] }, modelId: string) {
    this.maxActivation = data["maxActivation"] as number;

    // Construct samples
    const maxActivation = data["maxActivation"] as number;
    const sampleTexts = data["samples"] as string[];
    const samplesDecodedSamples = data["decodedTokens"] as string[][];
    const targetIdxs = data["tokenIdxs"] as number[];
    const absoluteTokenIdxs = data["absoluteTokenIdxs"] as number[];
    const sampleMagnitudeIdxs = data["magnitudeIdxs"] as number[][];
    const sampleMagnitudeValues = data["magnitudeValues"] as number[][];
    for (let i = 0; i < sampleTexts.length; i++) {
      const sampleText = sampleTexts[i];
      const decodedTokens = samplesDecodedSamples[i];
      const targetIdx = targetIdxs[i];
      const absoluteTokenIdx = absoluteTokenIdxs[i];
      // Build activations array from sparse representation
      const magnitudeIdxs = sampleMagnitudeIdxs[i];
      const magnitudeValues = sampleMagnitudeValues[i];
      const activations = Array(decodedTokens.length).fill(0);
      for (let j = 0; j < magnitudeIdxs.length; j++) {
        const idx = magnitudeIdxs[j];
        const value = magnitudeValues[j];
        activations[idx] = value;
      }
      const normalizedActivations = activations.map((a) => a / maxActivation);
      this.samples.push(
        new SampleData(
          sampleText,
          decodedTokens,
          activations,
          normalizedActivations,
          targetIdx,
          absoluteTokenIdx,
          modelId
        )
      );
    }
  }
}

// Creates a block profile data atom for a specific feature
function createBlockProfileAtom(block: BlockData) {
  return atomWithQuery((get) => ({
    // TODO: Replace with query for real data
    queryKey: [
      "predictions-data",
      get(modelIdAtom),
      get(sampleIdAtom),
      get(versionAtom),
      block.key,
    ],
    queryFn: async ({ queryKey: [, modelId, sampleId, version, blockKey] }) => {
      const url = `${SAMPLES_ROOT_URL}/${modelId}/samples/${sampleId}/${version}/${blockKey}.json`;
      const res = await fetch(url);
      const data = await res.json();
      return new BlockProfile(data, modelId as string);
    },
    staleTime: Infinity,
  }));
}

export { BlockData, BlockFeatureData, blocksAtom, createBlockModifierAtom, createBlockProfileAtom };
