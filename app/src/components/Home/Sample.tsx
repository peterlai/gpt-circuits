import classNames from "classnames";
import { getInspectSamplePath } from "../../views/App/urls";
import "./Sample.scss";

function SampleAnchor({
  modelId,
  sampleId,
  version,
  selectionKey,
  text,
}: {
  modelId: string;
  sampleId: string;
  version: string;
  selectionKey?: string;
  text?: string;
}) {
  const url = `#${getInspectSamplePath(modelId, sampleId, version, selectionKey)}`;

  return (
    <a href={url} target="child">
      {text ? text : `Sample ${sampleId}`}
    </a>
  );
}

function SampleQuote({
  text,
  targetIdx,
  targetLength,
  ellipsis = true,
  quotes = true,
}: {
  text: string;
  targetIdx: number;
  targetLength?: number;
  ellipsis?: boolean;
  quotes?: boolean;
}) {
  const targetEndIdx = targetLength ? targetIdx + targetLength : targetIdx + 1;

  return (
    <q
      className={classNames({
        "sample-quote": true,
        "without-quotes": !quotes,
      })}
    >
      {ellipsis && <>…</>}
      <span className="before">{text.slice(0, targetIdx)}</span>
      <em>{text.slice(targetIdx, targetEndIdx)}</em>
      <span className="after">{text.slice(targetEndIdx)}</span>
      {ellipsis && <>…</>}
    </q>
  );
}

function FeatureAnchor({
  modelId,
  sampleId,
  version,
  selectionKey,
  compact = false,
}: {
  modelId: string;
  sampleId: string;
  version: string;
  selectionKey: string;
  compact?: boolean;
}) {
  const url = `#${getInspectSamplePath(modelId, sampleId, version, selectionKey)}`;
  const featureLayer = selectionKey.split(".")[1];
  const featureId = selectionKey.split(".")[2];

  return (
    <a href={url} target="child">
      {!compact && <>Feature </>}
      {featureLayer}.{featureId}
    </a>
  );
}

export { FeatureAnchor, SampleAnchor, SampleQuote };
