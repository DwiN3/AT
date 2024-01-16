using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class showDieInfo : MonoBehaviour
{
    bool showInfo = false;

    // Start is called before the first frame update
    void Start()
    {
        // Pocz¹tkowo ukryj obiekt na podstawie wartoœci showInfo
        SetObjectVisibility();
    }

    // Update is called once per frame
    void Update()
    {
        // Przyk³adowe warunki zmiany wartoœci showInfo (mo¿esz dostosowaæ je wed³ug potrzeb)
        if (Input.GetKeyDown(KeyCode.Space))
        {
            showInfo = !showInfo;
            SetObjectVisibility();
        }
    }

    void SetObjectVisibility()
    {
        // SprawdŸ, czy obiekt ma komponent Collider (lub inny komponent, który móg³by go renderowaæ)
        Collider collider = GetComponent<Collider>();

        // Jeœli collider istnieje, mo¿esz ukryæ obiekt przez ustawienie aktywnoœci komponentu
        if (collider != null)
        {
            collider.enabled = showInfo;
        }
        else
        {
            // Jeœli collider nie istnieje, u¿yj aktywnoœci samego obiektu
            gameObject.SetActive(showInfo);
        }
    }
}
