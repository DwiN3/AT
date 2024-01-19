import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QuizSummaryComponent } from './quiz-summary.component';

describe('QuizSummaryComponent', () => {
  let component: QuizSummaryComponent;
  let fixture: ComponentFixture<QuizSummaryComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [QuizSummaryComponent]
    });
    fixture = TestBed.createComponent(QuizSummaryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
