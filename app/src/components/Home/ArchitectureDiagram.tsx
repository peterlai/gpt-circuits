import { Tooltip } from "react-tooltip";

import "./ArchitectureDiagram.scss";

function ArchitectureDiagram() {
  return (
    <figure id="ArchitectureDiagram">
      <figcaption>Autoencoder Locations</figcaption>
      <ol>
        <li>
          <div className="input">Inputs</div>
        </li>
        <li>
          <div className="block embedding">Embedding</div>
          <div
            className="sae"
            data-tooltip-id="DiagramTooltip"
            data-tooltip-content="Represents embeddings"
          >
            Sparse Autoencoder
          </div>
        </li>
        <li>
          <div className="block transformer">Transformer Block</div>
          <div
            className="sae"
            data-tooltip-id="DiagramTooltip"
            data-tooltip-content="Represents residual stream after first transformer block"
          >
            Sparse Autoencoder
          </div>
        </li>
        <li>
          <div className="block ellipsis">⋮</div>
          <div
            className="sae"
            data-tooltip-id="DiagramTooltip"
            data-tooltip-content="Represents residual stream before last transformer block"
          >
            Sparse Autoencoder
          </div>
        </li>
        <li>
          <div className="block transformer">Transformer Block</div>
          <div
            className="sae"
            data-tooltip-id="DiagramTooltip"
            data-tooltip-content="Represents residual stream before output probabilities are computed"
          >
            Sparse Autoencoder
          </div>
        </li>
        <li>
          <div className="block transposed-embedding">
            Embedding<sup>T</sup>
          </div>
        </li>
        <li>
          <div className="block softmax">Softmax</div>
        </li>
        <li>
          <div className="output">Output Probabilities</div>
        </li>
      </ol>
      <Tooltip id="DiagramTooltip" />
    </figure>
  );
}

export { ArchitectureDiagram };
