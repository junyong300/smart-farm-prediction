export class PestPredictResultDto {
  name: string;
  scores: {deviceIdx: number, score: number}[];
}