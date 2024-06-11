package pl.edu.atar.domain.dto;

import java.util.ArrayList;
import java.util.List;

public class ResponseData {
    private final List<String> invalidFieldNames;
    private int errorCode;

    public ResponseData() {
        this.invalidFieldNames = new ArrayList<>();
        this.errorCode = 200;
    }

    public void addInvalidFieldName(final String name) {
        invalidFieldNames.add(name);
        this.errorCode = 400;
    }

    public void addInvalidFieldNames(final List<String> fields) {
        if(!fields.isEmpty()) {
            invalidFieldNames.addAll(fields);
            this.errorCode = 400;
        }
    }

    public int getErrorCode() {
        return this.errorCode;
    }

    public String getInvalidFieldNames() {
        return String.join(",", this.invalidFieldNames);
    }
}
