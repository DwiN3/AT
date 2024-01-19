import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { BaseFlashcardInterface } from 'src/app/shared/models/flashcard.interface';
import { faTrash, faEdit } from '@fortawesome/free-solid-svg-icons';

@Component({
  selector: 'app-flashcard',
  templateUrl: './flashcard.component.html',
  styleUrls: ['./flashcard.component.scss']
})
export class FlashcardComponent
{
  @Output() onDeleteEvent = new EventEmitter<number>();
  @Output() onEditEvent = new EventEmitter<number>();
  @Input() flashcard : BaseFlashcardInterface | null = null;

  isFlipped: boolean = false;

  faTrash = faTrash;
  faEdit = faEdit;

  ToggleCard() 
  {
    this.isFlipped = !this.isFlipped;
  }

  OnDelete(id : number) : void
  {
    this.onDeleteEvent.emit(id);
  }

  OnEdit(id : number) : void
  {
    this.onEditEvent.emit(id);
  }

}
