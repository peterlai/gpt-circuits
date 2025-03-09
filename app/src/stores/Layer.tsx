import { atom } from "jotai";
import { sampleDataAtom } from "./Graph";

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

const layerProfilesAtom = atom<Record<number, LayerProfile>>((get) => {
  const sampleData = get(sampleDataAtom).data;
  if (sampleData && sampleData.klds) {
    const layerProfiles: Record<number, LayerProfile> = {};

    Object.keys(sampleData.klds).forEach((layerIdxStr) => {
      const layerIdx = parseInt(layerIdxStr);
      const kld = sampleData.klds[layerIdxStr] as number;
      const probabilities = sampleData.layerProbabilities[layerIdxStr] as { [key: string]: number };
      layerProfiles[layerIdx] = new LayerProfile(layerIdx, kld, probabilities);
    });
    return layerProfiles;
  } else {
    return {};
  }
});

export { LayerProfile, layerProfilesAtom };
