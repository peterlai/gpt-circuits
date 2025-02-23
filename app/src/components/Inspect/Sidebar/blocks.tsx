import { useAtomValue } from "jotai";
import { FaLayerGroup } from "react-icons/fa6";
import { BlockData } from "../../../stores/Block";
import { printableTokensAtom, targetIdxAtom } from "../../../stores/Graph";
import { CloseButton } from "./close";

function BlockSidebar({ block }: { block: BlockData }) {
  return (
    <>
      <BlockHeader block={block} />
    </>
  );
}

function BlockHeader({ block }: { block: BlockData }) {
  const printableTokens = useAtomValue(printableTokensAtom);
  const targetIdx = useAtomValue(targetIdxAtom);
  const blockToken = printableTokens[targetIdx - block.tokenOffset];

  return (
    <header>
      <span className="feature-location">
        <span>Examples like</span>&nbsp;
        <pre className="token">{blockToken}</pre>&nbsp;
        <span>on</span>
        <span className="layer">
          <FaLayerGroup className="icon" />
          {block.layerIdx > 0 ? `${block.layerIdx}` : "Embedding"}
        </span>
      </span>
      <CloseButton />
    </header>
  );
}

export { BlockSidebar };
