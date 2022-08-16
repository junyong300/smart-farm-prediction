import { Entity, Column } from 'typeorm';
import { SdhBase } from './sdh-base.entity';

@Entity({ name: 'sdh_external' })
export class SdhExternal extends SdhBase {
  @Column({ name: 'SEWS_TEMP', type: 'double' })
  temp?: number;
  @Column({ name: 'SEWS_HUMIDITY', type: 'double' })
  humidity?: number;
  @Column({ name: 'SEWS_WIND_DIRECTION' })
  windDirection?: string;
  @Column({ name: 'SEWS_WIND_SPEED', type: 'double' })
  windSpeed?: number;
  @Column({ name: 'SEWS_SOLAR_RADIATION', type: 'double' })
  solarRadiation?: number;
  @Column({ name: 'SEWS_LIGHTNESS', type: 'double' })
  lightness?: number;

  @Column({ name: 'SEWS_RAINFALL', type: 'double' })
  rainFall?: number;
  @Column({ name: 'SEWS_RAIN_FLAG' })
  rain?: boolean;

  fill() {
    // pass through
  }

  validate() {
    if (this.temp == null) return false;
    if (this.temp > 80 || this.temp < -50) return false;
    if (this.humidity != null && (this.humidity > 100 || this.humidity < 0)) return false;
    if (this.temp == 0 && this.humidity == 0) return false;

    return true;
  }
}
