import { atom } from "jotai";
import { selectAtom } from "jotai/utils";

import { sampleDataAtom } from "./Graph";
import { FeatureSelections, featureSelectionsAtom } from "./Selection";

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
  const ablationGraph = data?.ablation_graph ?? {};
  const activations: { [key: string]: number } = data?.activations ?? {};
  const normalizedActivations: { [key: string]: number } = data?.normalizedActivations ?? {};
  const groupAblationGraph: { [key: string]: [string, number][] } =
    data?.group_ablation_graph ?? {};

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

// Block background color options
enum BlockColorOptions {
  Default = 1,
  Blue = 2,
}

// Represents modifiers to apply to a specific block
class BlockModifier {
  public isHovered: boolean = false;
  public color: BlockColorOptions = BlockColorOptions.Default;

  constructor(block: BlockData, featureSelections: FeatureSelections) {
    // Show the hover state if the block is at the hovered upstream offset
    this.isHovered =
      featureSelections.focusedFeature?.layerIdx === block.layerIdx + 1 &&
      featureSelections.hoveredUpstreamOffset === block.tokenOffset;
  }

  // Used to avoid re-rendering when the modifier hasn't changed
  static areEqual(a: BlockModifier, b: BlockModifier) {
    return a.isHovered === b.isHovered && a.color === b.color;
  }
}

// Creates a block modifier atom for a specific block
function createBlockModifierAtom(block: BlockData) {
  return selectAtom(
    featureSelectionsAtom,
    (featureSelections) => {
      return new BlockModifier(block, featureSelections);
    },
    BlockModifier.areEqual
  );
}

export { BlockData, BlockFeatureData, blocksAtom, createBlockModifierAtom };
