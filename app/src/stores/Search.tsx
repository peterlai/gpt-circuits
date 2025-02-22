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

export { alignmentAtom, AlignmentOptions, searchQueryAtom };
