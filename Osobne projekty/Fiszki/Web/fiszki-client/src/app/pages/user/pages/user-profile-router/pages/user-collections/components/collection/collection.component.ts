import { Component, EventEmitter, Input, Output, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { faTrash, faEdit } from '@fortawesome/free-solid-svg-icons';
import { Store } from '@ngrx/store';
import { UserCollectionService } from 'src/app/pages/user/services/user-collection.service';
import { AlertModel } from 'src/app/shared/models/alert.model';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';
import { changeCollectionQuantity } from '../../store/carousel.actions';
import { CarouselState } from '../../store/carousel.state';
import { deleteCollection } from '../../store/collections.actions';
import { CollectionsState } from '../../store/collections.state';

@Component({
  selector: 'app-collection',
  templateUrl: './collection.component.html',
  styleUrls: ['./collection.component.scss']
})
export class CollectionComponent {

    @Input() collectionName : string = '';
    @Input() flashcardsQuantity : number = 0;
    @Output() alertEvent = new EventEmitter<AlertModel>();
    
    @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;

    faTrash = faTrash;
    faEdit = faEdit;

    alertDetails : AlertModel | null = null;

    constructor(private router : Router, private store : Store<{collections : CollectionsState}>, private carouselStore : Store<{carousel : CarouselState}>, private collectionService : UserCollectionService){}

    OnDelete() : void
    {
        this.collectionService.DeleteCollection(this.collectionName)
          .subscribe(data =>{
              this.store.dispatch(deleteCollection({name : this.collectionName}));
              this.carouselStore.dispatch(changeCollectionQuantity({value : -1}));
          }, err => {
            this.alertDetails = new AlertModel('Błąd serwera!', err.message, 'spróbuj ponownie później!');
            this.ChildShowAlert(this.alertDetails);
          })
    }

    private ChildShowAlert(alertDetails : AlertModel)
    {
      this.alertEvent.emit(alertDetails);
    }

    OnEdit() : void
    {
      this.router.navigate(['/user/profile/collections/edit'], { queryParams: { collectionName: this.collectionName }});
    }

}
