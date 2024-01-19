import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FlashcardEditFormComponent } from './flashcard-edit-form.component';

describe('FlashcardEditFormComponent', () => {
  let component: FlashcardEditFormComponent;
  let fixture: ComponentFixture<FlashcardEditFormComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [FlashcardEditFormComponent]
    });
    fixture = TestBed.createComponent(FlashcardEditFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
