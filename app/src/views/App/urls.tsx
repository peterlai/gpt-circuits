const SAMPLES_ROOT_URL = `${process.env.PUBLIC_URL}/samples`;

function getInspectSamplePath(modelId: string, sampleId: string, featureKey?: string) {
  if (featureKey && featureKey !== "") {
    return `samples/${modelId}/${sampleId}/${featureKey}`;
  } else {
    return `samples/${modelId}/${sampleId}`;
  }
}

function getEmbeddedSamplePath(modelId: string, sampleId: string, featureKey?: string) {
  const samplePath = getInspectSamplePath(modelId, sampleId, featureKey);
  return samplePath.replace("samples", "embedded");
}

function getAboutPath() {
  return "/";
}

function getV1Path() {
  return "/v1";
}

export { SAMPLES_ROOT_URL, getAboutPath, getEmbeddedSamplePath, getInspectSamplePath, getV1Path };
