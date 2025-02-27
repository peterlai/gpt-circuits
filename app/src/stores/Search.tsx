import { atom } from "jotai";

// Text search query
const searchQueryAtom = atom("");

// Alignment options
enum AlignmentOptions {
  Right = 1,
  Token = 2,
  Left = 3,
}
const alignmentAtom = atom(AlignmentOptions.Token);

// Sampling strategies
enum SamplingStrategies {
  Top = 1,
  Similar = 2,
  Cluster = 3,
}

const samplingStrategyAtom = atom(SamplingStrategies.Cluster);

export {
  alignmentAtom,
  AlignmentOptions,
  SamplingStrategies,
  samplingStrategyAtom,
  searchQueryAtom,
};
