// Represents data for a specific sample
class SampleData {
  public text: string;
  // Tokens decoded independently. If using tiktoken, concatenating these may not equal text
  public decodedTokens: string[];
  public tokens: SampleTokenData[] = [];
  public targetIdx: number;
  public absoluteTokenIdx: number;
  public modelId: string;

  constructor(
    text: string,
    decodedTokens: string[],
    activations: number[],
    normalizedActivation: number[],
    targetIdx: number,
    absoluteTokenIdx: number,
    modelId: string
  ) {
    this.text = text;
    this.decodedTokens = decodedTokens;
    this.targetIdx = targetIdx;
    this.absoluteTokenIdx = absoluteTokenIdx;
    this.modelId = modelId;

    // Create token data
    decodedTokens.forEach((value, index) => {
      this.tokens.push(
        new SampleTokenData(value, index, activations[index], normalizedActivation[index])
      );
    });
  }
}

class SampleTokenData {
  public value: string;
  public index: number;
  public activation: number;
  public normalizedActivation: number;

  // Replaces new lines with a special character for display
  get printableValue() {
    return this.value.replaceAll("\n", "⏎");
  }

  constructor(value: string, index: number, activation: number, normalizedActivation: number) {
    this.value = value;
    this.index = index;
    this.activation = activation;
    this.normalizedActivation = normalizedActivation;
  }
}

export { SampleData, SampleTokenData };
