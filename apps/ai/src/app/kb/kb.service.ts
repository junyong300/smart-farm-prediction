import { Injectable } from "@nestjs/common";
import { PestRuleDto } from "@libs/models/ai";

@Injectable()
export class KbService {
  constructor() {
    this.test();
  }

  test() {

    const t = `{
      "name": "흰곰팡이병",
      "period": 14,
      "conditions": [
        {"type": "crop", "value": 1, "score": 1},
        {"type": "date", "from": "03-01", "to": "06-01", "score": 5},
        {"type": "date", "from": "09-01", "to": "11-01", "score": 4},
        {"type": "internal", "field": "dryBulbTemp", "from": 20, "to": 25, "score": 5},
        {"type": "internal", "field": "humidity", "aggr": "avg", "from": 0, "to": 60, "score": 5},
        {"type": "internal", "field": "humidity", "aggr": "avg", "from": 0, "to": 60, "score": 5}
      ]
    }`;

    const tc:PestRuleDto = JSON.parse(t);
    // const tc = plainToClass(PestDto, obj);
    console.log(tc.constructor.name);
    console.log(tc.conditions[0].constructor.name);
    console.log(tc.conditions[3].constructor.name);
    console.log(tc.conditions[3].from);
    console.log(tc.conditions[3].field);
  }

  /**
   * 특정 날짜, 특정 농장의 병충해룰 점수
   * @param baseDate 
   */
  aa(baseDate = new Date()) {
    let startDate;
    let endDate;

  }

}