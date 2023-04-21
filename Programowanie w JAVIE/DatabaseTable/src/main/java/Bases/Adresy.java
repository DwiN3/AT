package Bases;

public class Adresy {
    private String street;
    private String nrhouse;
    private String locality;
    private String zipcode;

    public Adresy(String street, String nrhouse, String locality, String zipcode) {
        this.street = street;
        this.nrhouse = nrhouse;
        this.locality = locality;
        this.zipcode = zipcode;
    }

    public String getStreet() { return street; }
    public String getNrhouse() { return nrhouse; }
    public String getLocality() { return locality; }
    public String getZipcode() { return zipcode;}
}
