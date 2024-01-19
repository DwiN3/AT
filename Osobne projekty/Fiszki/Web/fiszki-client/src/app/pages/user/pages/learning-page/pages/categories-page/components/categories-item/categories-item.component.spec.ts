import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CategoriesItemComponent } from './categories-item.component';

describe('CategoriesItemComponent', () => {
  let component: CategoriesItemComponent;
  let fixture: ComponentFixture<CategoriesItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [CategoriesItemComponent]
    });
    fixture = TestBed.createComponent(CategoriesItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
