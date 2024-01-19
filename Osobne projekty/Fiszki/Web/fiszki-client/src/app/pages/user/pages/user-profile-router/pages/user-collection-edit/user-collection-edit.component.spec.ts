import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UserCollectionEditComponent } from './user-collection-edit.component';

describe('UserCollectionEditComponent', () => {
  let component: UserCollectionEditComponent;
  let fixture: ComponentFixture<UserCollectionEditComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [UserCollectionEditComponent]
    });
    fixture = TestBed.createComponent(UserCollectionEditComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
