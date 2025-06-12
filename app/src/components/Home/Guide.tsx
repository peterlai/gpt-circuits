import { getEmbeddedSamplePath } from "../../views/App/urls";
import "./Guide.scss";
import { SampleAnchor } from "./Sample";

function Guide() {
  const modelId = "toy-v0";
  const sampleId = "val.0.1024.15";
  const version = "0.15";
  const selectionKey = "2.2.170";
  const samplePath = `#${getEmbeddedSamplePath(modelId, sampleId, version, selectionKey)}`;

  return (
    <section id="Guide">
      <h2>How to Interpret a Circuit</h2>
      <p>
        Each column in the graph represents the processing of a different input token. Each row
        represents a different layer within the LLM. The first row represents features contained
        within the embedding layer. The last row represents features responsible for a final output
        prediction. Our interface supports tracing the downstream effects of “upstream” features.
        Select a feature to view its dependencies and list example sequences that trigger its
        activation.
      </p>
      <figure>
        <figcaption>
          Upstream Dependencies through{" "}
          <SampleAnchor
            modelId={modelId}
            sampleId={sampleId}
            version={version}
            selectionKey={selectionKey}
            text={`Feature #${selectionKey.split(".")[1]}.${selectionKey.split(".")[2]}`}
          />
        </figcaption>
        <a href={samplePath} target="_blank" rel="noopener noreferrer">
          <iframe src={samplePath} title="Embedded Sample" width="100%" height="500"></iframe>
        </a>
      </figure>
      <ul className="list">
        <li>
          <div>
            <span className="label">
              <span className="feature-number">123</span> :
            </span>
            <span className="description">
              Feature number. In this document, features may be prefixed with a layer number (e.g.,
              2.170).
            </span>
          </div>
        </li>
        <li>
          <div>
            <span className="label">
              <span className="lighter-feature">237</span> :
            </span>
            <span className="description">
              Lighter features are less important than bolded features.
            </span>
          </div>
        </li>
        <li>
          <div>
            <span className="label">
              <span className="highlighted-feature">170</span> :
            </span>
            <span className="description">Highlighted feature</span>
          </div>
        </li>
        <li>
          <div>
            <span className="label">
              <span className="dependent-feature">754</span> :
            </span>
            <span className="description">
              A feature that either depends upon or is dependent on the highlighted feature
            </span>
          </div>
        </li>
        <li>
          <div>
            <span className="label">
              <span className="weakly-iteracting-feature">34</span> :
            </span>
            <span className="description">
              A lighter background indicates a weaker feature interaction (i.e., muted ablation
              effect).
            </span>
          </div>
        </li>
      </ul>
    </section>
  );
}

export { Guide };
