import classNames from "classnames";
import { useAtom } from "jotai";
import { MdInfoOutline } from "react-icons/md";
import { TbLayoutAlignCenter, TbLayoutAlignLeft, TbLayoutAlignRight } from "react-icons/tb";
import { Tooltip } from "react-tooltip";

import { BlockFeatureData } from "../../../stores/Block";
import { HistogramData } from "../../../stores/Feature";
import { SampleData } from "../../../stores/Sample";
import { alignmentAtom, AlignmentOptions, searchQueryAtom } from "../../../stores/Search";
import { SamplesList } from "../../SamplesList";

import { JSX } from "react";
import "react-tooltip/dist/react-tooltip.css";

function SearchableSamples({
  samples,
  feature,
  activationHistogram,
  titleComponent,
}: {
  samples: SampleData[];
  feature?: BlockFeatureData;
  activationHistogram?: HistogramData;
  titleComponent?: JSX.Element;
}) {
  const [searchQuery, setSearchQuery] = useAtom(searchQueryAtom);
  const [alignment, setAlignment] = useAtom(alignmentAtom);

  // Function to check if sample should be shown based on search query.
  function filterSamples(sample: SampleData) {
    if (searchQuery === "") {
      return true;
    }

    // Escape special characters in search query for regex (except *?[])
    const escapedSearchQuery = searchQuery.replace(/[.+^${}()|\\]/g, "\\$&");

    // Replace * with .+, and ? with .
    const searchRegex = escapedSearchQuery.replace(/\*/g, ".+").replace(/\?/g, ".");

    // NOTE: using independently-decoded tokens here so won't necessarily be able to search
    //       based on the raw, original text. Also, will need to spell out byte encodings literally.
    const beforeText = sample.tokens
      .slice(0, sample.targetIdx)
      .map((t) => t.value)
      .join("");
    const targetText = sample.tokens[sample.targetIdx].value;
    const afterText = sample.tokens
      .slice(sample.targetIdx + 1)
      .map((t) => t.value)
      .join("");

    let searchableText: string;
    let targetTokenIdxs: [number, number];

    // If a | exists within the search query, add a | to the sample text after the target token.
    if (searchQuery.includes("|")) {
      // Search query must include the inserted "|"
      searchableText = beforeText + targetText + "|" + afterText;
      targetTokenIdxs = [
        beforeText.length + targetText.length,
        beforeText.length + targetText.length,
      ];
    } else {
      // Search query must include any character within the target token
      searchableText = beforeText + targetText + afterText;
      targetTokenIdxs = [beforeText.length, beforeText.length + targetText.length - 1];
    }

    try {
      // Check that at least one index corresponding to the target token is matched
      let regExp = new RegExp(searchRegex, "g");
      for (let match of searchableText.matchAll(regExp)) {
        const [matchStart, matchEnd] = [match.index, match.index + match[0].length - 1];
        if (matchStart <= targetTokenIdxs[1] && matchEnd >= targetTokenIdxs[0]) {
          return true;
        }
      }
    } catch (error) {
      console.error("Invalid regular expression:", error);
    }

    return false;
  }

  // Update filtered samples
  const filteredSamples = samples.filter(filterSamples);

  function prettyCount(count: number) {
    if (count > 1000000) {
      return `${(count / 1000000).toFixed(1)}M`;
    } else if (count > 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    } else {
      return count.toLocaleString();
    }
  }

  return (
    <>
      <header className="samples-header">
        <div className="info">
          {titleComponent ? titleComponent : null}
          <span className="count">
            {filteredSamples.length}{" "}
            {activationHistogram && <>/ {prettyCount(activationHistogram.totalCount)}</>}
          </span>
        </div>
        {activationHistogram && (
          <SamplesHistogram
            activationHistogram={activationHistogram}
            filteredSamples={filteredSamples}
          />
        )}
      </header>
      <section className="options">
        <span className="search">
          <input
            type="text"
            placeholder="Search text..."
            autoComplete="off"
            spellCheck="false"
            autoCorrect="off"
            onInput={(e) => setSearchQuery((e.target as HTMLInputElement).value)}
          />
          <MdInfoOutline className="icon" data-tooltip-id="SearchTooltip" />
          <Tooltip id="SearchTooltip">
            <code>* </code>: Match multiple letters
            <br />
            <code>? </code>: Match a single letter
            <br />
            <code>[]</code>: Match specific letters
            <br />
            <code>| </code>: Target token boundary
          </Tooltip>
        </span>
        <button
          title="Align left"
          className={classNames({
            selected: alignment === AlignmentOptions.Left,
          })}
          onClick={() => setAlignment(AlignmentOptions.Left)}
        >
          <TbLayoutAlignLeft className="icon" />
        </button>
        <button
          title="Align on target token"
          className={classNames({
            selected: alignment === AlignmentOptions.Token,
          })}
          onClick={() => setAlignment(AlignmentOptions.Token)}
        >
          <TbLayoutAlignCenter className="icon" />
        </button>
        <button
          title="Align right"
          className={classNames({
            selected: alignment === AlignmentOptions.Right,
          })}
          onClick={() => setAlignment(AlignmentOptions.Right)}
        >
          <TbLayoutAlignRight className="icon" />
        </button>
      </section>
      <section className="filtered-samples">
        <SamplesList samples={filteredSamples} alignment={alignment} />
      </section>
    </>
  );
}

function SamplesHistogram({
  activationHistogram,
  filteredSamples,
}: {
  activationHistogram: HistogramData;
  filteredSamples: SampleData[];
}) {
  // Count the filtered samples using activation histogram bins
  const sortedActivations = filteredSamples
    .map((sample) => sample.tokens[sample.targetIdx].activation)
    .sort((a, b) => a - b)
    .reverse();
  const binCounts = activationHistogram.bins.map(({ min, max }, i) => {
    let count = 0;
    while (sortedActivations) {
      const activation = sortedActivations.pop() ?? Infinity;
      if (activation <= max) {
        count++;
      } else {
        sortedActivations.push(activation);
        break;
      }
    }
    return count;
  });

  return (
    <div className="histogram">
      <table className="charts-css column show-primary-axis show-labels data-spacing-1">
        <tbody>
          {activationHistogram.bins.map(({ count, min, max }, i) => (
            <tr key={i}>
              {i === 0 && <th className="min">{min.toFixed(1)}</th>}
              {i === activationHistogram.bins.length - 1 && (
                <th className="max">{max.toFixed(1)}</th>
              )}
              <td
                style={
                  {
                    "--size": count / activationHistogram.maxCount,
                  } as React.CSSProperties
                }
                className={classNames({
                  represented: binCounts[i] > 0,
                })}
              ></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export { SearchableSamples };
