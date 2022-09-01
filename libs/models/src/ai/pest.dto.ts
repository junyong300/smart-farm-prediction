export class PestRuleCondition {
  type: "crop" | "date" | "internal" | "external" | "culture";
  field?: string;
  dayNight?: "all" | "day" | "night";
  aggr?: "avg" | "min" | "max"
  from?: string|number;
  to?: string|number;
  value?: string|number;
  score: number;
}

export class PestRuleDto {
  name: string;
  period: number;
  conditions: PestRuleCondition[]
}