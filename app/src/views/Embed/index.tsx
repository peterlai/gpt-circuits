import { useSetAtom } from "jotai";
import { useEffect } from "react";
import { useParams } from "react-router-dom";

import { AblationMap } from "../../components/Inspect/AblationMap";
import { modelIdAtom, sampleIdAtom, versionAtom } from "../../stores/Graph";

import "./style.scss";

function Embed() {
  const {
    modelId: modelIdFromUrl,
    sampleId: sampleIdFromUrl,
    version: versionFromUrl,
    selectionKey: selectionKeyFromUrl,
  } = useParams();
  const setSampleId = useSetAtom(sampleIdAtom);
  const setModelId = useSetAtom(modelIdAtom);
  const setVersion = useSetAtom(versionAtom);

  // Set sample ID and feature from URL params.
  useEffect(() => {
    setModelId(modelIdFromUrl ?? "");
    setSampleId(sampleIdFromUrl ?? "");
    setVersion(versionFromUrl ?? "");
  }, [modelIdFromUrl, sampleIdFromUrl, versionFromUrl, setModelId, setSampleId, setVersion]);

  // Add body class.
  useEffect(() => {
    document.body.classList.toggle("embedded", true);
  }, []);

  return <AblationMap selectionKeyFromUrl={selectionKeyFromUrl} isEmbedded={true} />;
}

export default Embed;
