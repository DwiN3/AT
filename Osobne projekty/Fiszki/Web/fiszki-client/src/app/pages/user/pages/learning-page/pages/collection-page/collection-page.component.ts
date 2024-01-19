import { Component, OnInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { Store } from '@ngrx/store';
import { Observable } from 'rxjs';
import { UserCollectionService } from 'src/app/pages/user/services/user-collection.service';
import { CollectionFlashcardsInterface } from 'src/app/shared/models/collection-flashcards.interface';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';
import { AlertService } from 'src/app/shared/ui/alert/service/alert.service';
import { FlashcardCollectionModel } from '../../../user-profile-router/pages/user-collections/models/flashcard-collection.model';
import { setCollection } from '../../store/game-settings.action';
import { GameSettingsState } from '../../store/game.state';

@Component({
  selector: 'app-collection-page',
  templateUrl: './collection-page.component.html',
  styleUrls: ['./collection-page.component.scss']
})
export class CollectionPageComponent implements OnInit{

  @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;

  gameSettings$! : Observable<GameSettingsState>

  isLoading : boolean = true;
  collections : FlashcardCollectionModel[] = [];
  quizMode : boolean = false;

  constructor(private collectionService : UserCollectionService, private alertService : AlertService, private store : Store<{gameSettings : GameSettingsState}>, private router : Router){}
  
  ngOnInit(): void 
  {
    this.collectionService.GetCollections()
      .subscribe(data => {
        this.collections = data;
      }, err => {
        console.log(err);
      }, () => {
        this.isLoading = false;
      })

      this.gameSettings$ = this.store.select('gameSettings');
      this.gameSettings$
        .subscribe(data => {
          if(data.learningMode !== "learning")
            this.quizMode = true;
        })
  }

  SetCollection(index : number) : void
  {
    const collectionName = this.collections[index].collectionName;
    this.isLoading = true;

    this.collectionService.GetFlashCardsByCollection(collectionName)
      .subscribe(data => {
        const flashcards = data;
        if(flashcards.length === 0)
          this.alertService.ShowAlert('Brak fiszek!', 'Najpierw dodaj fiszki do kolekcji', '', this.alertHost);
        else if(flashcards.length > 0 && flashcards.length < 4 && this.quizMode === true)
          this.alertService.ShowAlert('Za mało fiszek!', 'Tryb quiz potrzebuje przynajmniej 4 fiszek w zestawie!', '', this.alertHost);
        else
        {
          this.store.dispatch(setCollection({collection : flashcards}))
          this.router.navigate(['user/learning/game'])
        }
      }, err => {
        this.alertService.ShowAlert('Błąd serwera!', err.message, 'Spróbuj ponownie później', this.alertHost);
      }, () => {
        this.isLoading = false;
      })
    
  }



}
