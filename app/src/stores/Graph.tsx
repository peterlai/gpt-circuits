import { atom } from "jotai";
import { atomWithQuery } from "jotai-tanstack-query";

import { SAMPLES_ROOT_URL } from "../views/App/urls";
import { blocksAtom } from "./Block";

// Sample identifiers (set by view)
const modelIdAtom = atom("");
const sampleIdAtom = atom("");
const versionAtom = atom("");

// Raw sample data
const sampleDataAtom = atomWithQuery((get) => ({
  queryKey: [get(modelIdAtom), "sample-data", get(sampleIdAtom), get(versionAtom)],
  queryFn: async ({ queryKey: [modelId, , sampleId, version] }) => {
    if (!modelId || !sampleId || !version) return null;
    const url = `${SAMPLES_ROOT_URL}/${modelId}/samples/${sampleId}/${version}/data.json`;
    const res = await fetch(url);
    return res.json();
  },
  retry: 1,
  staleTime: Infinity,
}));

// String representation of each token
const sampleTokensAtom = atom<string[]>((get) => {
  const decodedTokens = get(sampleDataAtom).data?.decodedTokens;
  if (decodedTokens !== undefined) return decodedTokens;
  // Backwards compatible with shakespeare token-per-char. Remove when all regenerated.
  const sampleText = get(sampleDataAtom).data?.text ?? "";
  return [...sampleText];
});

// Map dimensions
const contextLengthAtom = atom((get) => get(sampleTokensAtom).length);
const numLayersAtom = atom((get) => {
  // Return deepest layer idx + 1
  const blocks = Object.values(get(blocksAtom));
  return blocks ? Math.max(...blocks.map((block) => block.layerIdx)) + 1 : 0;
});

// Target index on sample text
const targetIdxAtom = atom((get) => get(sampleDataAtom).data?.targetIdx ?? 0);

// Sample tokens (printable characters)
const printableTokensAtom = atom<string[]>((get) => {
  const sampleTokens = get(sampleTokensAtom);
  return sampleTokens.map((token) => token.replaceAll("\n", "⏎"));
});

// Target token (printable characters)
const targetTokenAtom = atom((get) => {
  return get(printableTokensAtom)[get(targetIdxAtom)];
});

// Predicted next tokens
const probabilitiesAtom = atom<{ [key: string]: number }>(
  (get) => get(sampleDataAtom).data?.probabilities ?? {}
);

// Predicted next tokens (using x_reconstructed)
const circuitProbabilitiesAtom = atom<{ [key: string]: number }>(
  (get) => get(sampleDataAtom).data?.layerProbabilities?.[get(numLayersAtom) - 1] ?? {}
);

export {
  circuitProbabilitiesAtom,
  contextLengthAtom,
  modelIdAtom,
  numLayersAtom,
  printableTokensAtom,
  probabilitiesAtom,
  sampleDataAtom,
  sampleIdAtom,
  sampleTokensAtom,
  targetIdxAtom,
  targetTokenAtom,
  versionAtom,
};
