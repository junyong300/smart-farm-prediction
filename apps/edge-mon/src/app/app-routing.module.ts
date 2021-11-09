import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PredictComponent } from './components/predict/predict.component';
import { SensorComponent } from './components/sensor/sensor.component';

const appRoutes: Routes = [
  { path: '', redirectTo: '/sensor', pathMatch: 'full'},
  { path: 'sensor', component: SensorComponent },
  { path: 'predict', component: PredictComponent },
];

@NgModule({
  imports: [
    RouterModule.forRoot(
      appRoutes,
      {
        enableTracing: false, // <-- debugging purposes only
      }
    )
  ],
  exports: [
    RouterModule
  ]
})

export class AppRoutingModule { }