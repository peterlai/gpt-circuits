import { atom } from "jotai";
import { numLayersAtom, sampleDataAtom } from "./Graph";

// Represents supplementary data for a specific layer
class LayerProfile {
  public idx: number;
  public kld: number;
  public probabilities: { [key: string]: number };

  constructor(idx: number, kld: number, probabilities: { [key: string]: number }) {
    this.idx = idx;
    this.kld = kld;
    this.probabilities = probabilities;
  }
}

const layerProfilesAtom = atom<LayerProfile[]>((get) => {
  const sampleData = get(sampleDataAtom).data;
  const numLayers = get(numLayersAtom);
  if (numLayers && sampleData) {
    const layerProfiles: LayerProfile[] = [];
    for (let i = 0; i < numLayers; i++) {
      const kld = sampleData.klds[`${i}`] as number;
      const probabilities = sampleData.layerProbabilities[`${i}`] as { [key: string]: number };
      layerProfiles.push(new LayerProfile(i, kld, probabilities));
    }
    return layerProfiles;
  } else {
    return [];
  }
});

export { LayerProfile, layerProfilesAtom };
