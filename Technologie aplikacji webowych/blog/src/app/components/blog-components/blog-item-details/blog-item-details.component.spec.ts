import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BlogItemDetailsComponent } from './blog-item-details.component';

describe('BlogItemDetailsComponent', () => {
  let component: BlogItemDetailsComponent;
  let fixture: ComponentFixture<BlogItemDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BlogItemDetailsComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(BlogItemDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
