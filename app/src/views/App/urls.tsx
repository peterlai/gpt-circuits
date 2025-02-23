const SAMPLES_ROOT_URL = `${process.env.PUBLIC_URL}/samples`;

function getInspectSamplePath(modelId: string, sampleId: string, selectionKey?: string) {
  if (selectionKey && selectionKey !== "") {
    return `samples/${modelId}/${sampleId}/${selectionKey}`;
  } else {
    return `samples/${modelId}/${sampleId}`;
  }
}

function getEmbeddedSamplePath(modelId: string, sampleId: string, selectionKey?: string) {
  const samplePath = getInspectSamplePath(modelId, sampleId, selectionKey);
  return samplePath.replace("samples", "embedded");
}

function getAboutPath() {
  return "/";
}

function getV1Path() {
  return "/v1";
}

export { getAboutPath, getEmbeddedSamplePath, getInspectSamplePath, getV1Path, SAMPLES_ROOT_URL };
