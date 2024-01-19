import { Component, EventEmitter, Output } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-summary-box',
  templateUrl: './summary-box.component.html',
  styleUrls: ['./summary-box.component.scss']
})
export class SummaryBoxComponent {

  @Output() playAgainEvent = new EventEmitter<void>();

  constructor(private router : Router){}

  PlayAgain() : void
  {
    this.playAgainEvent.emit();
  }

  ToMenuNav() : void
  {
    this.router.navigate(['/user'])
  }

}
