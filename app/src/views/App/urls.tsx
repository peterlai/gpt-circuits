const SAMPLES_ROOT_URL = `${process.env.PUBLIC_URL}/samples`;

function getInspectSamplePath(
  modelId: string,
  sampleId: string,
  version: string,
  selectionKey?: string
) {
  if (selectionKey && selectionKey !== "") {
    return `samples/${modelId}/${sampleId}/${version}/${selectionKey}`;
  } else {
    return `samples/${modelId}/${sampleId}/${version}`;
  }
}

function getEmbeddedSamplePath(
  modelId: string,
  sampleId: string,
  version: string,
  selectionKey?: string
) {
  const samplePath = getInspectSamplePath(modelId, sampleId, version, selectionKey);
  return samplePath.replace("samples", "embedded");
}

function getAboutPath() {
  return "/";
}

function getV1Path() {
  return "/v1";
}

export { getAboutPath, getEmbeddedSamplePath, getInspectSamplePath, getV1Path, SAMPLES_ROOT_URL };
