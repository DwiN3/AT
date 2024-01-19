import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'app-alert',
  templateUrl: './alert.component.html',
  styleUrls: ['./alert.component.scss']
})
export class AlertComponent {

  @Input() title : string = '';
  @Input() instructions : string = '';
  @Input() details : string = '';
  @Output() close = new EventEmitter<void>();

  OnClose() : void
  {
    this.close.emit();
  }

}
