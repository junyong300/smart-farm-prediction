import { Component } from '@angular/core';
import { HttpService } from './services/http.service';
import { InternalDto } from '@lib/models/sensor';

@Component({
  selector: 'smart-farm-prediction-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent {
  hello$ = this.http.get<InternalDto[]>("getInternal", {key: 'value'});

  constructor(private http: HttpService) {}
}
