import { Component, ComponentFactoryResolver, EventEmitter, Output, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';
import { faRemove } from '@fortawesome/free-solid-svg-icons';
import { Store } from '@ngrx/store';
import { Subscription } from 'rxjs';
import { UserCollectionService } from 'src/app/pages/user/services/user-collection.service';
import { AlertModel } from 'src/app/shared/models/alert.model';
import { AlertComponent } from 'src/app/shared/ui/alert/alert.component';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';
import { AlertService } from 'src/app/shared/ui/alert/service/alert.service';
import { FlashcardCollectionModel } from '../../models/flashcard-collection.model';
import { changeCollectionQuantity } from '../../store/carousel.actions';
import { CarouselState } from '../../store/carousel.state';
import { addCollection } from '../../store/collections.actions';
import { CollectionsState } from '../../store/collections.state';

@Component({
  selector: 'app-add-collection',
  templateUrl: './add-collection.component.html',
  styleUrls: ['./add-collection.component.scss']
})
export class AddCollectionComponent {

  @ViewChild('form') form : NgForm | null = null;
  @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;

  @Output() closeForm = new EventEmitter<boolean>();

  faRemove = faRemove;
  collectionName : string = '';
  isLoading : boolean = false;

  constructor(private userCollectionService : UserCollectionService, private store : Store<{collections : CollectionsState}>, private carouselStore : Store<{carousel : CarouselState}>, private alertService : AlertService){}

  CloseForm() : void
  {
    this.closeForm.emit(false);
  }

  OnAdd(collectionName : string) : void
  {
    if(this.form && !this.form.valid)
      return

      this.isLoading = true;
    
    this.userCollectionService.AddCollection(collectionName)
      .subscribe(data => {
          const collection = new FlashcardCollectionModel(collectionName, 0);
          this.store.dispatch(addCollection({collection : collection}))
          this.carouselStore.dispatch(changeCollectionQuantity({value : 1}))
          this.isLoading = false;
      }, err => {
        this.isLoading = false;
        if(err.status === 400)
          this.alertService.ShowAlert('Błąd!', 'Kolejkca o danej nazwie już istnieje', 'zmień nazwę kolekcji', this.alertHost);
        else 
          this.alertService.ShowAlert('Błąd serwera!', err.message, 'spróbuj ponownie później!', this.alertHost);
      })
      
  }

}
