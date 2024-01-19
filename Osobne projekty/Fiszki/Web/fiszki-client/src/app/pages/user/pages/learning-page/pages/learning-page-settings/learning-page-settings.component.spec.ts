import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LearningPageSettingsComponent } from './learning-page-settings.component';

describe('LearningPageSettingsComponent', () => {
  let component: LearningPageSettingsComponent;
  let fixture: ComponentFixture<LearningPageSettingsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [LearningPageSettingsComponent]
    });
    fixture = TestBed.createComponent(LearningPageSettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
