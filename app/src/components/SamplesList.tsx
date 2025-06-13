import classNames from "classnames";
import { TbBlockquote } from "react-icons/tb";

import { SampleData, SampleTokenData } from "../stores/Sample";
import { AlignmentOptions } from "../stores/Search";

import "./SamplesList.scss";

function SamplesList({
  samples,
  alignment,
  rightPadding,
  hideActivation,
  showIcon,
}: {
  samples: SampleData[];
  alignment: AlignmentOptions;
  rightPadding?: number;
  hideActivation?: boolean;
  showIcon?: boolean;
}) {
  const sortedSamples = samples.sort(
    (a, b) => a.tokens[a.targetIdx].activation - b.tokens[b.targetIdx].activation
  );

  return (
    <ul className="samples">
      {sortedSamples.reverse().map((sample, i) => (
        <Sample
          key={i}
          sample={sample}
          alignment={alignment}
          rightPadding={rightPadding || 10}
          hideActivation={hideActivation || false}
          showIcon={showIcon || false}
        />
      ))}
    </ul>
  );
}

function Sample({
  sample,
  alignment,
  rightPadding,
  hideActivation,
  showIcon,
}: {
  sample: SampleData;
  alignment: AlignmentOptions;
  rightPadding: number;
  hideActivation: boolean;
  showIcon: boolean;
}) {
  // Modify tokens based on alignment
  let tokens: SampleTokenData[] = [];
  switch (alignment) {
    case AlignmentOptions.Left:
      // Include first 128 tokens
      tokens.push(...sample.tokens.slice(0, 128));
      break;
    case AlignmentOptions.Token:
      // Char length of decoded tokens
      function getTextLength(tokens: SampleTokenData[]): number {
        return tokens.map((t) => t.value.length).reduce((acc, x) => acc + x, 0);
      }

      // Adds tokens until joined length satisfies rightPadding
      tokens = sample.tokens.slice(0, sample.targetIdx + 1);
      const rightTokens = [];
      for (let i = sample.targetIdx + 1; i < sample.tokens.length; i++) {
        const remainingPadding: number = rightPadding - getTextLength(rightTokens);
        const token = sample.tokens[i];
        if (remainingPadding > 0) {
          if (token.value.length <= remainingPadding) {
            // Add unmodified token if it fits
            rightTokens.push(sample.tokens[i]);
          } else {
            // Add truncated token if it doesn't fit
            rightTokens.push(
              new SampleTokenData(
                token.value.slice(0, remainingPadding),
                token.index,
                token.activation,
                token.normalizedActivation
              )
            );
          }
        } else {
          break;
        }
      }

      // Fill with empty tokens if necessary
      const paddingDeficit = rightPadding - getTextLength(rightTokens);
      tokens.push(...rightTokens);
      for (let i = 0; i < paddingDeficit; i++) {
        tokens.push(new SampleTokenData(" ", (tokens.at(-1)?.index ?? 0) + 1, 0, 0));
      }

      // Clamp to last 128 tokens
      tokens = tokens.slice(-128);
      break;
    case AlignmentOptions.Right:
      // Include last 128 tokens
      tokens.push(...sample.tokens.slice(-128));
      break;
  }

  // How many tokens have been clipped from the right?
  const remainder = Math.max(0, sample.tokens.length - ((tokens.at(-1)?.index ?? 0) + 1));

  // Build sample fragments
  const fragments: SampleFragmentData[] = [];
  let fragment = null;
  for (const token of tokens) {
    // Each token with a non-zero activation belongs in its own fragment
    if (!fragment || token.activation > 0 || (fragment.activation > 0 && token.activation === 0)) {
      fragment = new SampleFragmentData(
        token.printableValue,
        token.activation,
        token.normalizedActivation,
        token.index === sample.targetIdx
      );
      fragments.push(fragment);
      // Append to current no-activation fragment
    } else {
      fragment.text += token.printableValue;
    }
  }

  return (
    <li className="sample" data-sample-id={sample.absoluteTokenIdx}>
      {showIcon && <TbBlockquote className="icon" />}
      {!hideActivation && alignment === AlignmentOptions.Token && (
        <span className="index">{sample.targetIdx}</span>
      )}
      <span
        className={classNames({
          tokens: true,
          "align-left": alignment === AlignmentOptions.Left,
          "align-token": alignment === AlignmentOptions.Token,
          "align-right": alignment === AlignmentOptions.Right,
        })}
      >
        <pre>
          {fragments.map((fragment, i) => (
            <SampleFragment key={i} fragment={fragment} />
          ))}
        </pre>
      </span>
      {alignment === AlignmentOptions.Token && (
        <span className="remainder">{remainder > 0 ? <>…</> : <>&nbsp;</>}</span>
      )}
      {!hideActivation && (
        <span className="activation">{sample.tokens[sample.targetIdx].activation.toFixed(3)}</span>
      )}
    </li>
  );
}

class SampleFragmentData {
  text: string;
  activation: number;
  normalizedActivation: number;
  isTarget: boolean;

  constructor(text: string, activation: number, normalizedActivation: number, isTarget: boolean) {
    this.text = text;
    this.activation = activation;
    this.normalizedActivation = normalizedActivation;
    this.isTarget = isTarget;
  }
}

function SampleFragment({ fragment }: { fragment: SampleFragmentData }) {
  // Interpolate between RGB colors based on a value between 0 and 1
  function interpolateColor(value: number, startColor: number[], endColor: number[]) {
    return startColor
      .map((start, index) => start + (endColor[index] - start) * value)
      .map(Math.round);
  }

  // Given a value between 0 and 1, return an RGB color from white to yellow to red
  function fireGradient(value: number) {
    value = Math.max(0, Math.min(value, 1)); // clamp to [0, 1]
    const white = [255, 255, 255];
    const yellow = [255, 255, 0];
    const red = [255, 0, 0];

    if (value < 0.5) {
      return interpolateColor(value * 2, white, yellow);
    } else {
      return interpolateColor((value - 0.5) * 2, yellow, red);
    }
  }

  // Given a normalized activation value, return a style
  function activationStyle(value: number) {
    return {
      borderBottomColor: value > 0 ? `rgb(${fireGradient(value).join(",")})` : "transparent",
    };
  }

  if (fragment.activation === 0) {
    return <>{fragment.text}</>;
  }

  return (
    <span
      title={fragment.activation.toFixed(3)}
      style={activationStyle(fragment.normalizedActivation)}
      className={classNames({
        target: fragment.isTarget,
      })}
    >
      {fragment.text}
    </span>
  );
}

export { SamplesList };
