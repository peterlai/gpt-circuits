import { atom } from "jotai";
import { atomWithQuery } from "jotai-tanstack-query";

import { SAMPLES_ROOT_URL } from "../views/App/urls";

// When should the UX be in "mobile" mode?
const MOBILE_THRESHOLD = 800;
function isMobile(): boolean {
  return document.body.clientWidth <= MOBILE_THRESHOLD;
}

// Show menu by default if the screen is wide enough
const isMenuOpenAtom = atom(!isMobile());

// Show sidebar by default if the screen is wide enough
const isSidebarOpenAtom = atom(!isMobile());

class SampleOption {
  public modelId: string;
  public id: string;
  public versions: string[];
  public text: string;
  public decodedTokens: string[];
  public targetIdx: number;

  constructor(
    modelId: string,
    id: string,
    versions: string[],
    text: string,
    decodedTokens: string[],
    targetIdx: number
  ) {
    this.modelId = modelId;
    this.id = id;
    this.versions = versions;
    this.text = text;
    this.decodedTokens = decodedTokens;
    this.targetIdx = targetIdx;
  }

  get defaultVersion(): string {
    return this.versions[0];
  }

  get layerCount(): number {
    return ModelOption.getLayerCount(this.modelId);
  }
}

class ModelOption {
  public id: string;
  public sampleOptions: { [key: string]: SampleOption };

  constructor(id: string, sampleOptions: SampleOption[]) {
    this.id = id;
    this.sampleOptions = {};
    for (const sampleOption of sampleOptions) {
      this.sampleOptions[sampleOption.id] = sampleOption;
    }
  }

  // How many layers does a model have?
  static getLayerCount(id: string): number {
    // TODO: Add layer count to metadata for each model.
    if (id.startsWith("toy") || id.startsWith("ablation")) {
      return 4;
    } else if (id.startsWith("baby")) {
      return 6;
    } else {
      // Add more models here as needed.
      return 0;
    }
  }
}

// Sample options grouped by model ID
const modelOptionsAtom = atomWithQuery((get) => ({
  queryKey: ["model-options"],
  queryFn: async () => {
    const res = await fetch(`${SAMPLES_ROOT_URL}/index.json`);
    const models = await res.json();
    const modelOptions: { [key: string]: ModelOption } = {};
    // Iterate over the models and samples to create the ModelOption objects
    for (const modelId in models) {
      const samples = models[modelId];
      modelOptions[modelId] = new ModelOption(
        modelId,
        samples.map(
          (sample: any) =>
            new SampleOption(
              modelId,
              sample.name,
              sample.versions,
              sample.text,
              sample.decodedTokens,
              sample.targetIdx
            )
        )
      );
    }
    return modelOptions;
  },
  retry: 1,
  staleTime: Infinity,
}));

export { isMenuOpenAtom, isMobile, isSidebarOpenAtom, ModelOption, modelOptionsAtom, SampleOption };
