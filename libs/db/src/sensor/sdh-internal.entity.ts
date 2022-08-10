import { Entity, Column } from 'typeorm';
import { SdhBase } from './sdh-base.entity';
import { AMPM_TYPE } from '@libs/utils';

@Entity({ name: 'sdh_internal' })
export class SdhInternal extends SdhBase {

  @Column({ name: 'SIE_TEMP', type: 'double', comment: '' })
  temp?: number;
  @Column({ name: 'SIE_HUMIDITY', type: 'double', comment: '' })
  humidity?: number;
  @Column({ name: 'SIE_WET_BULB_TEMP', type: 'double', comment: '' })
  wetBulbTemp?: number;
  @Column({ name: 'SIE_CO2', type: 'double', comment: 'co2' })
  co2?: number;
  @Column({ name: 'SIE_LIGHTNESS', type: 'double', comment: '' })
  lightness?: number;

  @Column({ name: 'SIE_SOLAR_RADIANTION', type: 'double', comment: '' })
  solarRadiation?: number;
  @Column({ name: 'SIE_ILLUMINANCE', type: 'double', comment: '' })
  illuminance?: number;
  @Column({ name: 'SIE_DEW_POINT_TEMP', type: 'double', comment: '' })
  dewPointTemp?: number;
  @Column({ name: 'SIE_HD', type: 'double', comment: '' })
  hd?: number;
  @Column({ name: 'SIE_HD_LEVEL', comment: '' })
  hdLevel?: number;
  @Column({ name: 'SIE_CM_TEMP', comment: '' })
  cultureMediumTemp?: number;
  @Column({ name: 'SIE_CM_HUMIDITY', comment: '' })
  cultureMediumHumidity?: number;
  @Column({ name: 'SIE_ABS_WATER', comment: '' })
  absoluteWaterContent?: number;

  @Column({ name: 'SIE_AM_PM_TYPE', type: 'varchar', comment: '' })
  amPmType?: AMPM_TYPE;
  @Column({ name: 'SIE_SUNRISE_SUNSET_FLAG', comment: '' })
  isSunrise?: boolean;

  @Column({ name: 'SIE_ENTHALPY', comment: '' })
  enthalpy?: number;

  @Column({ name: 'TIME_DIFF' })
  timeDiff?: number;

  fill() {

    if (isNaN(this.humidity)) this.humidity = null;

    if (isNaN(this.dewPointTemp)) this.dewPointTemp = null;

    if (isNaN(this.hd)) this.hd = null;

    this.amPmType = this.sensingDt.getHours() >= 12 ? AMPM_TYPE.PM : AMPM_TYPE.AM;
  }

  validate() {
    if (this.temp == null) return false;
    if (this.temp > 80 || this.temp < -50) return false;
    if (this.humidity != null && (this.humidity > 100 || this.humidity < 0)) return false;
    if (this.temp == 0 && this.humidity == 0) return false; 

    return true;
  }
}
