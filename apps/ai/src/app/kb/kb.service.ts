import { Injectable } from "@nestjs/common";
import { PestRuleCondition, PestRuleDto } from "@libs/models/ai";
import { InjectRepository } from "@nestjs/typeorm";
import { Repository } from "typeorm";
import { SdhBase, SdhInternal } from "@libs/db/sensor";
import { DateUtils } from "@libs/utils";

@Injectable()
export class KbService {
  constructor(
    @InjectRepository(SdhInternal) private sdhInternalRepo: Repository<SdhInternal> 
  ) {
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
        {"type": "internal", "field": "sie_temp", "from": 20, "to": 25, "score": 5},
        {"type": "internal", "field": "sie_humidity", "aggr": "avg", "from": 0, "to": 60, "score": 5},
        {"type": "internal", "field": "sie_humidity", "aggr": "avg", "from": 0, "to": 60, "score": 5}
      ]
    }`;

    const tc:PestRuleDto = JSON.parse(t);
    // const tc = plainToClass(PestDto, obj);
    this.aa(tc);
  }

  /**
   * 
   * @param baseDate 
   */
  async aa(rule: PestRuleDto, baseDate = new Date()) {
    let startDate = DateUtils.add(baseDate, -rule.period);
    let endDate = baseDate;
    const startDk = DateUtils.toDayKey(startDate);
    const endDk = DateUtils.toDayKey(endDate);

    for (const cond of rule.conditions) {
      await this.evalCondition(cond, startDk, endDk);
    }
  }

  evalCondition(cond: PestRuleCondition, startDk, endDk) {
    switch (cond.type) {
      case "internal" : 
      case "external" : 
      case "culture" : 
        this.evalEnvCondition(cond, startDk, endDk);
    }
  }

  async evalDateCondition(cond: PestRuleCondition, startDk, endDk) {
    
  }

  async evalEnvCondition(cond: PestRuleCondition, startDk, endDk, deviceIdxes?: []) {
    let repo: Repository<SdhBase>;

    switch(cond.type) {
      case "internal" : repo = this.sdhInternalRepo; break;
      case "external" : 
      case "culture" : 
    }

    let field = cond.aggr ? `${cond.aggr}(${cond.field})` : cond.field;

    let query = repo.createQueryBuilder()
      .select(["device_idx", "dayKey"])
      .addSelect(field, "value")
      .where("daykey between :startDk and :endDk", {startDk: startDk, endDk: endDk});

    if (deviceIdxes) {
      query = query.andWhere("device_idx in (:devices)", {devices: deviceIdxes.toString()});
    }

    query = query.groupBy("device_idx").addGroupBy("dayKey")
      
    let rs = await query.getRawMany();
    console.log(rs);

    const scores = new Map<number, number>();

    for (const r of rs) {
      if (cond.value != null && (+r.value == +cond.value)) {
        let score = (scores.get(+r.device_idx) || 0) + cond.score;
        scores.set(+r.device_idx, score);
      } else if(cond.from != null && cond.to != null && +r.value >= +cond.from && +r.value <= +cond.to) {
        let score = (scores.get(+r.device_idx) || 0) + cond.score;
        scores.set(+r.device_idx, score);
      }
    }

    console.log(scores);

    return scores;

  }

}