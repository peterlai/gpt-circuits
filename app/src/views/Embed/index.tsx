import { useSetAtom } from "jotai";
import { useEffect } from "react";
import { useParams } from "react-router-dom";

import { AblationMap } from "../../components/Inspect/AblationMap";
import { modelIdAtom, sampleIdAtom } from "../../stores/Graph";

import "./style.scss";

function Embed() {
  const {
    modelId: modelIdFromUrl,
    sampleId: sampleIdFromUrl,
    featureKey: featureKeyFromUrl,
  } = useParams();
  const setSampleId = useSetAtom(sampleIdAtom);
  const setModelId = useSetAtom(modelIdAtom);

  // Set sample ID and feature from URL params.
  useEffect(() => {
    setModelId(modelIdFromUrl ?? "");
    setSampleId(sampleIdFromUrl ?? "");
  }, [modelIdFromUrl, sampleIdFromUrl, setModelId, setSampleId]);

  // Add body class.
  useEffect(() => {
    document.body.classList.toggle("embedded", true);
  }, []);

  return <AblationMap featureKeyFromUrl={featureKeyFromUrl} isEmbedded={true} />;
}

export default Embed;
