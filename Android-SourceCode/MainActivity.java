package com.restaurantrating.rrating;

/*
MIT License

Copyright (c) 2020 Somnath Saha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

import android.content.pm.PackageManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.view.View;
import android.widget.Button;
import android.content.Intent;
import android.provider.MediaStore;
import android.graphics.Bitmap;
import android.util.Base64;
import android.app.ProgressDialog;
import android.widget.Toast;
import android.widget.EditText;



import org.json.JSONException;
import java.io.IOException;
import java.net.MalformedURLException;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.util.Map;
import java.util.Hashtable;
import java.util.HashMap;


import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.net.HttpURLConnection;

import com.android.volley.NetworkResponse;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.android.volley.toolbox.*;
import com.android.volley.AuthFailureError;


// JSON
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.*;






public class MainActivity extends AppCompatActivity {

    static final int REQUEST_IMAGE_CAPTURE = 1;
    ImageView restaurantImage;
    Button getRating;
    Bitmap imageBitmap;
    TextView txtViewRating;
    private EditText editTextName;
    private String KEY_IMAGE = "image";
    private String KEY_NAME  = "name";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnTakeImage = (Button) findViewById(R.id.btnTakeImage);
        restaurantImage     = (ImageView) findViewById(R.id.imgViewRestaurant);
        getRating           = (Button) findViewById(R.id.btnGetRating);
        txtViewRating       = (TextView) findViewById(R.id.txtViewRating);

        // Check the device has camera, else disable button
        if (! hasCamera())
        {
            btnTakeImage.setEnabled(false);
        }


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            // Get bitmap image
            imageBitmap = (Bitmap) extras.get("data");
            // Set bitmap image
            restaurantImage.setImageBitmap(imageBitmap);
        }



    }

    /*
  Function Name: launchCamera()
  Input Args   : <View>
  Return       : None
  Description  : Launch Camera ( Event Handler )
   */
    public void launchCamera( View view)
    {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }

    }

    /*
   Function Name: hasCamera()
   Input Args   : <View>
   Return       : boolean
   Description  : Check if the device has camera or not
    */
    private boolean hasCamera()
    {
        return getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);

    }

    /*
   Function Name: getRating()
   Input Args   : <View>
   Return       : None
   Description  : This function does the following jobs. ( Event Handler )
                  - Upload the captured image to the server to be processed
                  - Invoke showRating(<JSON>) function to process JSON response and display
                    Restaurant Name and Rating.
    */
    public void getRating(View view)
    {
        // URL for image to be uploaded
        String uploadURL = "http://10.0.1.32:5000/upload";
        // Upload Image to the above URL
        uploadImage(uploadURL);
    }

    /*
    Function Name: getStringImage()
    Input Args   : <Bitmap Image>
    Return       : <String: image in string format>
    Description  : Returns string image
    */
    public String getStringImage(Bitmap bmp){
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] imageBytes = baos.toByteArray();
        String encodedImage = Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;
    }

    /*
    Function Name: uploadImage()
    Input Args   : <Upload URL , API>
    Return       : None
    Description  : Upload the image to the upload dir of webserver.
                   - Display the success / Failure message
                   - Display Restaurant Name and Rating on Success.
     */
    private void uploadImage(String UPLOAD_URL){

        //Creating a Request Queue
        RequestQueue requestQueue = Volley.newRequestQueue(this);
        //Showing the progress dialog
        final ProgressDialog loading = ProgressDialog.show(this,"Uploading...","Please wait...",false,false);
        StringRequest stringRequest = new StringRequest(Request.Method.POST, UPLOAD_URL,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String serverResponse) {
                        //Dismissing the progress dialog
                        loading.dismiss();
                        //Showing toast message of the response
                        Toast.makeText(MainActivity.this, "Uploaded Successfully!" , Toast.LENGTH_LONG).show();
                        // Display Restaurant Name and Rating to the textview
                        txtViewRating.setText(serverResponse);
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError volleyError) {
                        //Dismissing the progress dialog
                        loading.dismiss();

                        //Showing toast
                        //Toast.makeText(MainActivity.this, volleyError.getMessage().toString(), Toast.LENGTH_LONG).show();
                        Toast.makeText(MainActivity.this, "Upload Failed!" , Toast.LENGTH_LONG).show();
                        txtViewRating.setText("ERROR MSG:" + volleyError.getMessage().toString());
                    }
                }) {
            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                //Converting Bitmap to String
                String image = getStringImage(imageBitmap);

                //Getting Image Name
                //String name = editTextName.getText().toString().trim();
                String name = "rr.png";
                //Creating parameters
                //Map<String, String> params = new Hashtable<String, String>();
                Map<String, String> params = new HashMap<>();

                //Adding parameters
                params.put(KEY_IMAGE, image);
                params.put(KEY_NAME, name);

                //returning parameters
                return params;
            }
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String,String> params = new HashMap<String, String>();
                // Removed this line if you dont need it or Use application/json
                // params.put("Content-Type", "application/x-www-form-urlencoded");
                return params;
            }
        };

        //Adding request to the queue
        requestQueue.add(stringRequest);
    }



}
