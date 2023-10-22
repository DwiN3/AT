using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DoorManager : MonoBehaviour
{
    bool doorIsOpen = false;
    float doorTimer = 0.0f;
    public float doorOpenTime = 3.0f;
    public AudioClip doorOpenSound;
    public AudioClip doorShutSound;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (doorIsOpen)
        {
            doorTimer += Time.deltaTime;
        }
        if (doorTimer > doorOpenTime)
        {
            Door(false, doorShutSound, "doorshut");
            doorTimer = 0.0f;
        }
    }

    void DoorCheck()
    {
        if (!doorIsOpen)
        {
            Door(true, doorOpenSound, "dooropen");
        }
    }

    void Door(bool openCheck, AudioClip aClip, string animName)
    {
        doorIsOpen = openCheck;
        GetComponent<AudioSource>().PlayOneShot(aClip);
        transform.parent.GetComponent<Animation>().Play(animName);
    }
}
