import { JSX } from "react";
import "./ProbabilitiesTable.scss";

function ProbabilitiesTable({
  probabilities,
  labels,
  limit = 4,
}: {
  probabilities: { [key: string]: number };
  labels?: { [key: string]: JSX.Element };
  limit?: number;
}) {
  const nonZeroProbabilities = Object.entries(probabilities).filter(([, p]) => p > 0.01);
  const sortedProbabilities = nonZeroProbabilities.sort(([, a], [, b]) => b - a);

  // Only show the top  probabilities
  const selectProbabilities = sortedProbabilities.slice(0, limit);
  const maxTokenLength = Math.max(...selectProbabilities.map(([token]) => token.length));

  // Create labels for top probabilities
  const rowLabels: { [key: string]: JSX.Element } = {};
  selectProbabilities.forEach(([token]) => {
    if (labels && labels[token]) {
      rowLabels[token] = labels[token];
    } else {
      // Replace new lines.
      const text = token.replace(/\n/g, "⏎");
      rowLabels[token] = <pre className="token">{text}</pre>;
    }
  });

  return (
    <table className="probabilities charts-css bar show-labels data-spacing-1">
      <tbody style={{ "--labels-size": `${maxTokenLength + 2.5}ch` } as React.CSSProperties}>
        {selectProbabilities.map(([token, probability]) => (
          <tr key={token} className="prediction">
            <th scope="row">{rowLabels[token]}</th>
            <td style={{ "--size": probability } as React.CSSProperties}>
              {probability > 0.1 ? (
                <span className="data"> {(probability * 100).toFixed(0)}% </span>
              ) : null}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export { ProbabilitiesTable };
