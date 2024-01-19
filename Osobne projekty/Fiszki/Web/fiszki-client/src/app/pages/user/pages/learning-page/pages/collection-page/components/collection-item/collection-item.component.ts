import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-collection-item',
  templateUrl: './collection-item.component.html',
  styleUrls: ['./collection-item.component.scss']
})
export class CollectionItemComponent {

  @Input() collectionName : string = '';
  @Input() flashcardsQuantity : number = 0;

}
