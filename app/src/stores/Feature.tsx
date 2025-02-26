import { atomWithQuery } from "jotai-tanstack-query";
import { selectAtom } from "jotai/utils";

import { SAMPLES_ROOT_URL } from "../views/App/urls";
import { BlockFeatureData } from "./Block";
import { modelIdAtom, sampleIdAtom, versionAtom } from "./Graph";
import { SampleData } from "./Sample";
import { SamplingStrategies, samplingStrategyAtom } from "./Search";
import { SelectionState, selectionStateAtom } from "./Selection";

// Feature text color options
enum TextColorOptions {
  Default = 1,
  Blue = 2,
}

// Represents modifiers to apply to a specific feature
class FeatureModifier {
  public isHovered: boolean = false;
  public isSelected: boolean = false;
  // True if the focused feature shares the same feature ID and is on the same layer
  public isRelatedToFocused: boolean = false;
  // True if an ajacent feature in the same block is focused
  public isNeighborFocused: boolean = false;
  public isActive: boolean = false;
  public isGray: boolean = false;
  public fillWeight: number = 0;
  public textWeight: number = 0;
  public textColor: TextColorOptions = TextColorOptions.Default;

  constructor(feature: BlockFeatureData, selectionState: SelectionState) {
    this.isHovered = feature.key === selectionState.hoveredFeature?.key;
    this.isSelected = feature.key === selectionState.selectedFeature?.key;

    // Set text weight using normalized activation
    if (feature.normalizedActivation < 0.5) {
      this.textWeight = 1;
    } else if (feature.normalizedActivation < 0.65) {
      this.textWeight = 2;
    } else {
      this.textWeight = 3;
    }

    if (!this.isHovered && !this.isSelected) {
      // Does the focused feature share the same feature ID and is on the same layer?
      const focusedFeature = selectionState.focusedFeature;
      if (
        focusedFeature &&
        focusedFeature.tokenOffset !== feature.tokenOffset &&
        focusedFeature.layerIdx === feature.layerIdx &&
        focusedFeature.featureId === feature.featureId
      ) {
        this.isRelatedToFocused = true;
      }

      // Check if an adjacent feature in the same block is focused
      if (
        focusedFeature &&
        focusedFeature.layerIdx === feature.layerIdx &&
        focusedFeature.tokenOffset === feature.tokenOffset &&
        // HACK: Using math to check if two features are next to each other
        (focusedFeature.featureId + feature.featureId) * 0.1 >
          Math.abs(focusedFeature.featureId - feature.featureId)
      ) {
        this.isNeighborFocused = true;
      }

      // Check if focused feature is upstream or downstream
      let ablationWeight: number = 0;
      if (focusedFeature) {
        // If focused feature is upstream, activate feature
        if (Object.keys(feature.ablatedBy).includes(focusedFeature.key)) {
          this.isActive = true;
          ablationWeight = feature.ablatedBy[focusedFeature.key].weight;
        }
        // If focused feature is downstream, activate feature
        if (Object.keys(focusedFeature.ablatedBy).includes(feature.key)) {
          this.isActive = true;
          ablationWeight = focusedFeature.ablatedBy[feature.key].weight;
        }
      }

      // If not active, check if selected feature is upstream or downstream
      const selectedFeature = selectionState.selectedFeature;
      if (!this.isActive && selectedFeature) {
        // If selected feature is upstream, activate feature
        if (Object.keys(feature.ablatedBy).includes(selectedFeature.key)) {
          this.isActive = true;
          this.isGray = true;
          ablationWeight = feature.ablatedBy[selectedFeature.key].weight;
        }
        // If selected feature is downstream, activate feature
        if (Object.keys(selectedFeature.ablatedBy).includes(feature.key)) {
          this.isActive = true;
          this.isGray = true;
          ablationWeight = selectedFeature.ablatedBy[feature.key].weight;
        }
      }

      // If ablation weight exists, set weight
      if (ablationWeight !== 0) {
        if (ablationWeight < 0.1) {
          this.fillWeight = 1;
        } else if (ablationWeight < 5.0) {
          this.fillWeight = 2;
        } else {
          this.fillWeight = 3;
        }
      }
    }
  }

  // Used to avoid re-rendering when the feature modifier hasn't changed
  static areEqual(a: FeatureModifier, b: FeatureModifier) {
    return (
      a.isHovered === b.isHovered &&
      a.isSelected === b.isSelected &&
      a.isActive === b.isActive &&
      a.fillWeight === b.fillWeight &&
      a.textWeight === b.textWeight &&
      a.textColor === b.textColor &&
      a.isGray === b.isGray &&
      a.isRelatedToFocused === b.isRelatedToFocused &&
      a.isNeighborFocused === b.isNeighborFocused
    );
  }
}

// Creates a feature modifier atom for a specific feature key
function createFeatureModifierAtom(feature: BlockFeatureData) {
  return selectAtom(
    selectionStateAtom,
    (selectionState) => {
      return new FeatureModifier(feature, selectionState);
    },
    FeatureModifier.areEqual
  );
}

// Represents supplementary data for a specific feature
class FeatureProfile {
  public maxActivation: number = 0;
  public samples: SampleData[] = [];
  public activationHistogram: HistogramData;

  constructor(data: { [key: string]: number | [] }, modelId: string) {
    this.maxActivation = data["maxActivation"] as number;
    this.activationHistogram = new HistogramData(data.activationHistogram as {});

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

// Represents histogram data
class HistogramData {
  public maxCount: number;
  public totalCount: number;
  public bins: HistogramBin[];

  constructor(data: { counts?: number[]; binEdges?: number[] }) {
    const counts = data.counts || [];
    const binEdges = data.binEdges || [];
    this.maxCount = Math.max(...counts);
    this.totalCount = counts.reduce((sum, count) => sum + count, 0);
    this.bins = counts.map(
      (count, idx) =>
        ({
          count,
          min: binEdges[idx],
          max: binEdges[idx + 1],
        } as HistogramBin)
    );

    // If there are fewer than 10 bins, add more bins (improves histogram appearance)
    while (this.bins.length > 0 && this.bins.length < 10) {
      const prevBin = this.bins[this.bins.length - 1];
      this.bins.push({ count: 0, min: prevBin.max, max: prevBin.max * 2 - prevBin.min });
    }
  }
}

type HistogramBin = {
  count: number;
  min: number;
  max: number;
};

// Creates a feature profile data atom for a specific feature
function createFeatureProfileAtom(feature: BlockFeatureData) {
  return atomWithQuery((get) => ({
    queryKey: [
      get(modelIdAtom),
      "feature-profile-data",
      get(sampleIdAtom),
      get(versionAtom),
      get(samplingStrategyAtom),
      feature.key,
    ],
    queryFn: async ({ queryKey: [modelId, , sampleId, version, samplingStrategy, featureKey] }) => {
      let url: string;
      switch (samplingStrategy) {
        case SamplingStrategies.Cluster:
          url = `${SAMPLES_ROOT_URL}/${modelId}/samples/${sampleId}/${version}/${featureKey}.json`;
          break;
        default:
          const [, layerIdx, featureIdx] = (featureKey as string).split(".");
          url = `${SAMPLES_ROOT_URL}/${modelId}/features/${layerIdx}.${featureIdx}.json`;
          break;
      }
      const res = await fetch(url);
      const data = await res.json();
      return new FeatureProfile(data, modelId as string);
    },
    staleTime: Infinity,
  }));
}

export { createFeatureModifierAtom, createFeatureProfileAtom, FeatureProfile, HistogramData };
