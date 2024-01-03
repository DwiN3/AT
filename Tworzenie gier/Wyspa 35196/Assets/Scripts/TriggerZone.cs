using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TriggerZone : MonoBehaviour
{
    public AudioClip lockedSound;
    public Light doorLight;
    public Text textHints;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (GameObject.Find("PowerGUI") && Inventory.charge == 4)
        {
            doorLight.color = Color.green;
        }
    }

    void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.tag == "Player")
        {
            if (Inventory.charge == 4)
            {
                Destroy(GameObject.Find("PowerGUI"));
                transform.Find("door").SendMessage("DoorCheck");
            }
            else if (Inventory.charge > 0 && Inventory.charge < 4)
            {
                textHints.SendMessage("ShowHint", "Drzwi ani drgną … \n pewnie potrzebują więcej mocy...");
                transform.Find("door").GetComponent<AudioSource>().PlayOneShot(lockedSound);
            }
            else
            {
                transform.Find("door").GetComponent<AudioSource>().PlayOneShot(lockedSound);
                col.gameObject.SendMessage("HUDon");
                textHints.SendMessage("ShowHint", "Te drzwi wyglądają na zamknięte,\n być może generator wymaga\n odpowiedniego zasilania...");
            }
        }
    }
}
