import { Entity, Column } from 'typeorm';
import { SdhBase } from './sdh-base.entity';

@Entity({ name: 'sdh_external' })
export class SdhExternal extends SdhBase {
  @Column({ name: 'SEWS_TEMP', type: 'double', comment: '온도(건구)' })
  temp?: number;
  @Column({ name: 'SEWS_HUMIDITY', type: 'double', comment: '습도' })
  humidity?: number;
  @Column({ name: 'SEWS_WIND_DIRECTION', comment: '풍향' })
  windDirection?: string;
  @Column({ name: 'SEWS_WIND_SPEED', type: 'double', comment: '풍속' })
  windSpeed?: number;
  @Column({ name: 'SEWS_SOLAR_RADIATION', type: 'double', comment: '일사' })
  solarRadiation?: number;
  @Column({ name: 'SEWS_LIGHTNESS', type: 'double', comment: '광량'})
  lightness?: number;

  @Column({ name: 'SEWS_RAINFALL', type: 'double', comment: '강우량' })
  rainFall?: number;
  @Column({ name: 'SEWS_RAIN_FLAG', comment: '감우' })
  rain?: boolean;

  fill() {
    // pass through
  }

  validate() {
    if (this.temp == null) return false;
    if (this.temp > 80 || this.temp < -50) return false;
    if (this.humidity != null && (this.humidity > 100 || this.humidity < 0)) return false;
    if (this.temp == 0 && this.humidity == 0) return false; // 둘 다 0이면 잘못된 것으로 간주

    return true;
  }
}
