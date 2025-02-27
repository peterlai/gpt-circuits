import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useEffect } from "react";
import { BsReverseLayoutTextSidebarReverse } from "react-icons/bs";
import { CgSpinner } from "react-icons/cg";
import { FaLayerGroup, FaX } from "react-icons/fa6";

import {
  modelIdAtom,
  sampleDataAtom,
  sampleIdAtom,
  sampleTokensAtom,
  targetIdxAtom,
  versionAtom,
} from "../stores/Graph";
import { isSidebarOpenAtom, ModelOption, modelOptionsAtom } from "../stores/Navigation";

import "./Navbar.scss";

function Navbar() {
  const setIsSidebarOpen = useSetAtom(isSidebarOpenAtom);
  const [{ isPending, isError }] = useAtom(sampleDataAtom);
  const modelId = useAtomValue(modelIdAtom);
  const sampleId = useAtomValue(sampleIdAtom);
  const version = useAtomValue(versionAtom);

  const Loader = () => (
    <span className="loading">
      <CgSpinner className="spinner spin" />
      Loading
    </span>
  );

  const Error = () => (
    <span className="error">
      <FaX className="icon" />
      <span>Couldn't find&nbsp;</span>
      <pre>
        {sampleId}:{version}
      </pre>
      <span>&nbsp;in&nbsp;</span>
      <pre>{modelId}</pre>
    </span>
  );

  return (
    <div id="Navbar">
      <div className="left-side">
        {isPending ? <Loader /> : isError ? <Error /> : <NavbarContent />}
      </div>
      <div className="right-side">
        {!isPending && !isError && <VersionSelector />}
        <BsReverseLayoutTextSidebarReverse
          id="ToggleSidebar"
          onClick={() => setIsSidebarOpen((isOpen) => !isOpen)}
        />
      </div>
    </div>
  );
}

function trimTokens(tokens: string[], targetIdx: number, leftChars: number, rightChars: number) {
  const leftTokens: string[] = [];
  for (let i = targetIdx - 1; i >= 0 && leftTokens.join("").length < leftChars; i--) {
    leftTokens.unshift(tokens[i]);
  }
  const rightTokens: string[] = [];
  for (let i = targetIdx + 1; i < tokens.length && rightTokens.join("").length < rightChars; i++) {
    rightTokens.push(tokens[i]);
  }
  const isTrimmedLeft = leftTokens.length < targetIdx;
  const isTrimmedRight = rightTokens.length < tokens.length - targetIdx - 1;
  return { leftTokens, rightTokens, isTrimmedLeft, isTrimmedRight };
}

function NavbarContent() {
  const modelId = useAtomValue(modelIdAtom);
  const sampleId = useAtomValue(sampleIdAtom);
  const version = useAtomValue(versionAtom);
  const sampleTokens = useAtomValue(sampleTokensAtom);
  const targetIdx = useAtomValue(targetIdxAtom);
  const layerCount = ModelOption.getLayerCount(modelId);

  // Replace newlines with a return symbol
  const printableTokens = sampleTokens.map((t) => t.replaceAll("\n", "⏎"));

  // Truncate the original context
  const { leftTokens, rightTokens, isTrimmedLeft, isTrimmedRight } = trimTokens(
    printableTokens,
    targetIdx,
    50,
    10
  );

  useEffect(() => {
    // Set document title
    const { leftTokens, rightTokens, isTrimmedLeft, isTrimmedRight } = trimTokens(
      printableTokens,
      targetIdx,
      10,
      10
    );
    const text = `${leftTokens.join("")}[${printableTokens[targetIdx]}]${rightTokens.join("")}`;
    const titleText = `${isTrimmedLeft ? "…" : ""}${text}${isTrimmedRight ? "…" : ""}`;
    document.title = `GPT Circuit | "${titleText}" | ${modelId}:${sampleId}:${version}`;
  }, [modelId, sampleId, version, targetIdx, printableTokens, layerCount]);

  return (
    <>
      <span className="identifier">
        <span className="layer-count">
          {layerCount ? (
            <>
              <FaLayerGroup className="icon" />
              {layerCount}
            </>
          ) : (
            modelId
          )}
        </span>
        <span className="sample-id">{sampleId}</span>
      </span>
      <span className="context">
        &ldquo;
        {isTrimmedLeft && <span className="ellipsis">...</span>}
        <span className="sample">
          {[...leftTokens, printableTokens[targetIdx], ...rightTokens].map((tokenStr, i) => (
            <span
              key={i}
              className={classNames({
                target: i === leftTokens.length,
                return: tokenStr === "⏎",
              })}
            >
              {tokenStr}
            </span>
          ))}
        </span>
        {isTrimmedRight && <span className="ellipsis">...</span>}
        &rdquo;
      </span>
    </>
  );
}

function VersionSelector() {
  const [{ data: modelOptions }] = useAtom(modelOptionsAtom);
  const setVersion = useSetAtom(versionAtom);
  const modelId = useAtomValue(modelIdAtom);
  const sampleId = useAtomValue(sampleIdAtom);
  const version = useAtomValue(versionAtom);

  // Find options for the current sample.
  const options = modelOptions?.[modelId]?.sampleOptions[sampleId]?.versions || [version];

  return (
    <select
      className="version-selector"
      value={version}
      onChange={(e) => {
        const newVersion = e.target.value;
        if (newVersion !== version) {
          // Update version to show
          setVersion(newVersion);
        }
      }}
    >
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

export { Navbar };
