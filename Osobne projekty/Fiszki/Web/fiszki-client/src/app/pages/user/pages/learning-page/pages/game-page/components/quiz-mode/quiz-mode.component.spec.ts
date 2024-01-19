import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QuizModeComponent } from './quiz-mode.component';

describe('QuizModeComponent', () => {
  let component: QuizModeComponent;
  let fixture: ComponentFixture<QuizModeComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [QuizModeComponent]
    });
    fixture = TestBed.createComponent(QuizModeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
