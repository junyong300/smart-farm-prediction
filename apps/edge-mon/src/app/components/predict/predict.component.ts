import { AfterViewInit, Component, ElementRef, NgZone, OnInit, ViewChild } from '@angular/core';
import { HttpService } from '../../services/http.service';
import { PredictInternalSelfDto } from '@libs/models/sensor';
import { MatTableDataSource } from '@angular/material/table';
import * as am5 from '@amcharts/amcharts5';
import am5themes_Animated from '@amcharts/amcharts5/themes/Animated';
import * as am5xy from '@amcharts/amcharts5/xy';

@Component({
  selector: 'predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.scss'],
})
export class PredictComponent implements OnInit, AfterViewInit {
  constructor(private zone: NgZone, private http: HttpService) {}

  ngOnInit() {}

  ngAfterViewInit() {
    this.http.get<PredictInternalSelfDto>('ml.serving', { model: 'InternalSelf' })
      .subscribe((res) => {

        let root = am5.Root.new('chartdiv');

        root.setThemes([am5themes_Animated.new(root)]);

        let chart = root.container.children.push(
          am5xy.XYChart.new(root, {
            panX: true,
            panY: true,
            wheelX: 'panX',
            wheelY: 'zoomX',
          })
        );

        let cursor = chart.set(
          'cursor',
          am5xy.XYCursor.new(root, {
            behavior: 'none',
          })
        );
        cursor.lineY.set('visible', false);

        let xAxis = chart.xAxes.push(
          am5xy.DateAxis.new(root, {
            maxDeviation: 0.5,
            baseInterval: {
              timeUnit: 'minute',
              count: 10,
            },
            gridIntervals: [
              { timeUnit: "minute", count: 10 },
              { timeUnit: "minute", count: 30 },
              { timeUnit: "hour", count: 1 },
              { timeUnit: "hour", count: 3 },
              { timeUnit: "hour", count: 6 },
              { timeUnit: "hour", count: 12 },
            ],
            renderer: am5xy.AxisRendererX.new(root, {}),
            tooltip: am5.Tooltip.new(root, {}),
          })
        );

        let yAxis = chart.yAxes.push(
          am5xy.ValueAxis.new(root, {
            renderer: am5xy.AxisRendererY.new(root, {}),
          })
        );

        const colorSet = am5.ColorSet.new(root, {});

        let seriesInput = chart.series.push(
          am5xy.LineSeries.new(root, {
            name: 'SeriesInput',
            xAxis: xAxis,
            yAxis: yAxis,
            valueYField: 'value',
            valueXField: 'date',
            stroke: colorSet.getIndex(0),
            tooltip: am5.Tooltip.new(root, {
              labelText: '{valueY}',
            }),
          })
        );


        let seriesPredict = chart.series.push(
          am5xy.LineSeries.new(root, {
            name: 'SeriesPredict',
            xAxis: xAxis,
            yAxis: yAxis,
            valueYField: 'value',
            valueXField: 'date',
            stroke: colorSet.getIndex(6),
            tooltip: am5.Tooltip.new(root, {
              labelText: '{valueY}',
            }),
          })
        );


        chart.set(
          'scrollbarX',
          am5.Scrollbar.new(root, {
            orientation: 'horizontal',
          })
        );


        const dataInput: any[] = [];
        for (let i = 0; i < res.input.length; i++) {
          console.log(res.input[i]);
          const date = new Date(String(res.input[i][0]));
          dataInput.push({
            date: date.getTime(),
            value: res.input[i][1],
          })
        }

        const dataPredict: any[] = [];
        let date = new Date(String(res.input[res.input.length - 1][0]));
        for (let i = 0; i < res.pred.length; i++) {
          am5.time.add(date, "minute", 10);
          dataPredict.push({
            date: date.getTime(),
            value: Math.round(res.pred[i] * 10) / 10,
          })
        }

        seriesInput.data.setAll(dataInput);
        seriesPredict.data.setAll(dataPredict);

        seriesInput.appear(100);
        chart.appear(100, 10);
      });
  }
}
