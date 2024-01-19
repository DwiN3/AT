import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CollectionRouterOutletComponent } from './collection-router-outlet.component';

describe('CollectionRouterOutletComponent', () => {
  let component: CollectionRouterOutletComponent;
  let fixture: ComponentFixture<CollectionRouterOutletComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [CollectionRouterOutletComponent]
    });
    fixture = TestBed.createComponent(CollectionRouterOutletComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
