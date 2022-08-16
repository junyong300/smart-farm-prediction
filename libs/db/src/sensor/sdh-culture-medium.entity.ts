import { AMPM_TYPE } from '@libs/utils';
import { Entity, Column } from 'typeorm';
import { SdhBase } from './sdh-base.entity';

@Entity({ name: 'sdh_culture_medium' })
export class SdhCultureMedium extends SdhBase {
  @Column({ name: 'AM_PM_TYPE', comment: 'am or pm' })
  amPmType?: string;
  @Column({ name: 'SUNRISE_SUNSET_FLAG', comment: 'is day or night' })
  isSunrise?: boolean;

  @Column({ name: 'SCM_TEMP', type: 'double' })
  temp?: number;          
  @Column({ name: 'SCM_MOISTURE_CONTENT', type: 'double' })
  moistureContent?: number;            
  @Column({ name: 'SCM_EC', type: 'double',  comment: 'EC value'})
  ec?: number;
  @Column({ name: 'SCM_PH', type: 'double',  comment: 'PH value'})
  ph?: number;

  fill() {
    this.amPmType = this.sensingDt.getHours() >= 12 ? AMPM_TYPE.PM : AMPM_TYPE.AM;
  }

  validate() {
    if (!this.temp && !this.moistureContent && !this.ec) return false;
    return true;
  }
}
