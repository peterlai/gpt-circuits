import { atomWithQuery } from "jotai-tanstack-query";

import { SAMPLES_ROOT_URL } from "../views/App/urls";
import { modelIdAtom, numLayersAtom, sampleIdAtom } from "./Graph";
import { SampleData } from "./Sample";

// Represents similar samples to a given sample.
class PredictionData {
  public samples: SampleData[] = [];
  public nextTokenProbabilities: { [key: string]: number } = {};

  constructor(data: { [key: string]: number | [] }, modelId: string) {
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

    // Find and count the tokens afters the target token.
    const tokenCounts: { [key: string]: number } = {};
    for (const sample of this.samples) {
      if (sample.targetIdx === sample.tokens.length - 1) {
        continue;
      }
      const token = sample.tokens[sample.targetIdx + 1];
      if (token.value in tokenCounts) {
        tokenCounts[token.value] += 1;
      } else {
        tokenCounts[token.value] = 1;
      }
    }

    // Construct circuit probabilities
    const numTokens = Object.values(tokenCounts).reduce((a, b) => a + b, 0);
    for (const [tokenValue, count] of Object.entries(tokenCounts)) {
      this.nextTokenProbabilities[tokenValue] = count / numTokens;
    }
  }
}

// Prediction data, which includes similar samples to a given sample.
const predictionDataAtom = atomWithQuery((get) => ({
  queryKey: ["predictions-data", get(modelIdAtom), get(sampleIdAtom), get(numLayersAtom) - 1],
  queryFn: async ({ queryKey: [, modelId, sampleId, layerIdx] }) => {
    if (!modelId || !sampleId || Number(layerIdx) < 0) return null;
    const res = await fetch(
      `${SAMPLES_ROOT_URL}/${modelId}/samples/${sampleId}/0.${layerIdx}.json`
    );
    const data = await res.json();
    return new PredictionData(data, modelId as string);
  },
  retry: 1,
  staleTime: Infinity,
}));

export { PredictionData, predictionDataAtom };
